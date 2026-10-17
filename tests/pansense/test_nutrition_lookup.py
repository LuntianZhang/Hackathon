from copy import deepcopy
from io import BytesIO
from pathlib import Path
import threading
import time
from urllib.error import HTTPError

from fastapi.testclient import TestClient
import pytest

from pansense.api import create_app
from pansense.nutrition import Catalog
from pansense.nutrition_lookup import NutritionLookup, PublicNutritionProvider, valid_nutrients
from pansense.store import Store


RESULT = {'status': 'found', 'result': {
    'per_100g': {'kcal': 211, 'protein': 19.4, 'carbs': 12.7, 'fat': 11},
    'source': 'USDA FoodData Central', 'source_url': 'https://fdc.nal.usda.gov/food-details/172443/nutrients',
    'source_id': '172443', 'reference_name': 'Natto', 'source_description': 'Natto',
    'quality': 'official_reference'}}


def wait_for(lookup, identifier):
    until = time.monotonic() + 5
    while time.monotonic() < until:
        value = lookup.get(identifier)
        if value['status'] not in {'queued', 'searching'}: return value
        time.sleep(.01)
    pytest.fail('Background query did not finish')


def store_at(root):
    return Store(root, Catalog(Path('app/catalog/foods.json')))


def test_background_search_is_nonblocking_deduplicated_cached_and_does_not_change_revision(tmp_path):
    store = store_at(tmp_path)
    entered, release = threading.Event(), threading.Event()
    calls = []
    def provider(name, query):
        calls.append((name, query)); entered.set(); assert release.wait(5)
        return deepcopy(RESULT)
    lookup = NutritionLookup(store, provider=provider); store.nutrition_lookup = lookup
    try:
        meal = store.create('manual')
        meal = store.add_item(meal['id'], 'Natto', None, meal['revision'])
        assert entered.wait(2)
        item = meal['items'][0]
        assert item['nutrition'] is None and item['nutrition_lookup']['status'] in {'queued', 'searching'}
        identifier = item['nutrition_lookup']['id']
        assert lookup.start('  NATTO ')['id'] == identifier
        release.set(); assert wait_for(lookup, identifier)['status'] == 'found'
        enriched = store.get(meal['id'])
        assert enriched['revision'] == meal['revision']
        assert enriched['totals']['kcal'] == 211 and enriched['totals']['unknown_items'] == 0
        assert enriched['items'][0]['nutrition_needs_review']
        assert enriched['items'][0]['nutrition']['source_id'] == '172443'
        assert lookup.start('Natto')['status'] == 'found' and len(calls) == 1
    finally:
        release.set(); lookup.close()
    reopened = store_at(tmp_path)
    cache = NutritionLookup(reopened, enabled=False); reopened.nutrition_lookup = cache
    try: assert reopened.get(meal['id'])['totals']['kcal'] == 211
    finally: cache.close()


def test_late_query_cannot_overwrite_edit_or_recreate_deleted_meal(tmp_path):
    store = store_at(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def provider(*_):
        entered.set(); assert release.wait(5); return deepcopy(RESULT)
    lookup = NutritionLookup(store, provider=provider); store.nutrition_lookup = lookup
    try:
        meal = store.create('manual')
        meal = store.add_item(meal['id'], 'Natto', None, meal['revision'])
        assert entered.wait(2)
        identifier = meal['items'][0]['nutrition_lookup']['id']
        edited = store.edit_item(meal['id'], meal['items'][0]['id'], '牛奶', 'milk', meal['revision'])
        second = store.create('manual', items=[store.new_item('Natto')])
        store.delete(second['id'])
        release.set(); wait_for(lookup, identifier)
        assert store.get(meal['id'])['items'][0]['name'] == '牛奶'
        assert store.get(meal['id'])['totals']['kcal'] == 61
        assert store.get(meal['id'])['revision'] == edited['revision']
        assert not store.exists(second['id'])
    finally:
        release.set(); lookup.close()


def test_purge_discards_inflight_cache_and_nonfood_never_queries(tmp_path):
    store = store_at(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def provider(*_): entered.set(); assert release.wait(5); return deepcopy(RESULT)
    lookup = NutritionLookup(store, provider=provider); store.nutrition_lookup = lookup
    try:
        store.create('photo', items=[store.new_item('鼠标', object_kind='non_food'),
                                    store.new_item('饼干包装', object_kind='packaged_food')])
        assert not lookup.futures
        request = lookup.start('Natto'); assert entered.wait(2)
        store.purge(); release.set()
        lookup.pool.shutdown(wait=True)
        with pytest.raises(KeyError): lookup.get(request['id'])
        assert not store.list(include_drafts=True)
    finally:
        release.set(); lookup.close()


@pytest.mark.parametrize('status', ['not_found', 'ambiguous', 'error'])
def test_unavailable_and_retry_never_invent_zero_nutrients(tmp_path, status):
    store = store_at(tmp_path); calls = []
    def provider(*_): calls.append(1); return {'status': status} if len(calls) == 1 else deepcopy(RESULT)
    lookup = NutritionLookup(store, provider=provider)
    try:
        first = lookup.start('Natto'); value = wait_for(lookup, first['id'])
        assert value['status'] == status and value['nutrition'] is None
        assert lookup.start('Natto')['status'] == status and len(calls) == 1
        lookup.start('Natto', retry=True)
        assert wait_for(lookup, first['id'])['status'] == 'found' and len(calls) == 2
    finally: lookup.close()


def test_restart_does_not_leave_queries_pending_forever(tmp_path):
    store = store_at(tmp_path); first = NutritionLookup(store, enabled=False)
    request = first.start('新食品'); first.close()
    second = NutritionLookup(store, enabled=False)
    try:
        assert second.get(request['id'])['status'] == 'error'
        assert second.start('新食品')['status'] == 'queued'
    finally: second.close()


def test_usda_requires_exact_food_and_complete_per_100g_units(monkeypatch):
    provider = PublicNutritionProvider()
    nutrients = [{'nutrientId': code, 'unitName': unit, 'value': value}
                 for code, unit, value in [(1008, 'KCAL', 211), (1003, 'G', 19.4), (1004, 'G', 11), (1005, 'G', 12.7)]]
    foods = [{'fdcId': 1, 'description': 'Natto', 'foodNutrients': nutrients},
             {'fdcId': 2, 'description': 'Natto sauce', 'foodNutrients': nutrients},
             {'fdcId': 3, 'description': 'Natto', 'foodNutrients': nutrients[1:]}]
    monkeypatch.setattr(provider, '_json', lambda *_: {'foods': foods})
    result = provider.usda('Natto')
    assert len(result) == 1 and result[0]['per_100g']['kcal'] == 211
    assert result[0]['source_id'] == '1'


def test_product_labels_reject_ml_missing_fields_and_different_recipe(monkeypatch):
    provider = PublicNutritionProvider()
    product = {'code': '12345678', 'product_name': '测试食品', 'nutrition_data_per': '100g',
               'nutriments': {'energy-kj_100g': 418.4, 'proteins_100g': 10, 'carbohydrates_100g': 10, 'fat_100g': 2}}
    ml = {**product, 'nutrition_data_per': '100ml'}
    incomplete = {**product, 'nutriments': {'energy-kcal_100g': 100}}
    recipe = {**product, 'product_name': '测试食品酱'}
    monkeypatch.setattr(provider, '_json', lambda *_: {'products': [product, ml, incomplete, recipe]})
    result = provider.open_food_facts('测试食品')
    assert len(result) == 1 and result[0]['per_100g']['kcal'] == pytest.approx(100)
    assert result[0]['license'] == 'ODbL'


def test_different_brand_nutrition_is_ambiguous_not_arbitrarily_selected(monkeypatch):
    provider = PublicNutritionProvider()
    one = deepcopy(RESULT['result']); two = deepcopy(one); two['per_100g']['kcal'] = 250
    monkeypatch.setattr(provider, 'usda', lambda _: [])
    monkeypatch.setattr(provider, 'open_food_facts', lambda _: [one, two])
    assert provider('食品', 'food')['status'] == 'ambiguous'


def test_temporary_upstream_outage_retries_once_without_leaking_query_url(monkeypatch):
    provider = PublicNutritionProvider(); calls = []
    def read(*_, **__):
        calls.append(1)
        if len(calls) == 1: raise HTTPError('https://world.openfoodfacts.org/',503,'Unavailable',{},None)
        return BytesIO(b'{"products":[]}')
    monkeypatch.setattr('pansense.nutrition_lookup.urlopen', read)
    monkeypatch.setattr(provider.stop, 'wait', lambda _: False)
    assert provider._json('https://world.openfoodfacts.org/cgi/search.pl', {'search_terms': 'food'}) == {'products': []}
    assert len(calls) == 2


@pytest.mark.parametrize('values', [
    {'kcal': 100, 'protein': None, 'fat': 1, 'carbs': 10},
    {'kcal': float('nan'), 'protein': 1, 'fat': 1, 'carbs': 10},
    {'kcal': 100, 'protein': 80, 'fat': 80, 'carbs': 80},
])
def test_invalid_source_values_are_not_nutrition(values): assert not valid_nutrients(values)


def test_api_manual_unknown_food_automatically_queries_and_known_food_is_local(tmp_path):
    entered, release = threading.Event(), threading.Event()
    def provider(*_): entered.set(); assert release.wait(5); return deepcopy(RESULT)
    app = create_app(Path.cwd(), tmp_path, jobs_enabled=False, nutrition_enabled=True, nutrition_provider=provider)
    try:
        with TestClient(app) as client:
            local = client.post('/api/nutrition/lookups', json={'name': '牛奶'}).json()
            assert local['status'] == 'found' and local['nutrition']['kcal'] == 61
            assert not entered.is_set()
            meal = client.post('/api/meals/manual').json()
            response = client.post(f"/api/meals/{meal['id']}/foods", json={'name': 'Natto', 'revision': meal['revision']})
            assert response.status_code == 201
            meal = response.json(); request = meal['items'][0]['nutrition_lookup']
            assert entered.wait(2)
            assert client.get('/api/nutrition/lookups/' + request['id']).json()['status'] in {'queued', 'searching'}
            assert meal['totals']['unknown_items'] == 1
            release.set(); wait_for(app.state.nutrition_lookup, request['id'])
            updated = client.get('/api/meals/' + meal['id']).json()
            assert updated['totals']['kcal'] == 211 and updated['revision'] == meal['revision']
            assert updated['items'][0]['nutrition']['source_url'].startswith('https://fdc.nal.usda.gov/')
    finally: release.set()
