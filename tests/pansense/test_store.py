from pathlib import Path

import pytest

from pansense.nutrition import Catalog
from pansense.store import ConflictError, Store

@pytest.fixture
def store(tmp_path):
    return Store(tmp_path, Catalog(Path('app/catalog/foods.json')))

def add(store, meal, key):
    return store.add_item(meal['id'], store.catalog.foods[key]['name'], key, meal['revision'])

def test_corrections_recalculate_and_unknown_food_is_not_zero_calories(store):
    meal = add(store, store.create('manual'), 'banana')
    assert meal['totals']['kcal'] == pytest.approx(89*1.18)
    old_revision = meal['revision']
    meal = store.edit_item(meal['id'], meal['items'][0]['id'], '米饭', 'rice', old_revision)
    assert meal['totals']['kcal'] == 195
    with pytest.raises(ConflictError):
        store.delete_item(meal['id'], meal['items'][0]['id'], old_revision)
    meal = store.add_item(meal['id'], '自制未知酱汁', None, meal['revision'])
    assert meal['items'][-1]['nutrition'] is None
    assert meal['totals']['unknown_items'] == 1
    assert not meal['totals']['complete']

def test_persistence_idempotent_save_and_local_date_boundary(store):
    meal = add(store, store.create('manual'), 'egg')
    meal = store.save(meal['id'], {'occurred_at': '2026-10-11T16:30:00.000Z', 'meal_type': 'breakfast'}, meal['revision'])
    meal = store.save(meal['id'], {}, meal['revision'])
    assert len(store.list('2026-10-12')) == 1
    assert store.list('2026-10-11') == []
    reopened = Store(store.root, store.catalog)
    assert reopened.summary('2026-10-12')['count'] == 1
    assert reopened.summary('2026-10-12')['totals']['kcal'] == pytest.approx(77.5)
    with pytest.raises(ValueError):
        store.save(meal['id'], {'occurred_at': '2026-10-12T12:00'}, meal['revision'])

def test_delete_and_privacy_reset_clear_records_and_preferences(store):
    first = add(store, store.create('manual'), 'apple')
    second = add(store, store.create('manual'), 'rice')
    store.delete(first['id'])
    with pytest.raises(KeyError):
        store.get(first['id'])
    store.update_profile({'onboarded': True, 'goal_kcal': 2000})
    store.purge()
    assert store.list(include_drafts=True) == []
    assert store.profile()['goal_kcal'] is None

def test_empty_draft_cannot_save_and_catalog_has_traceable_sources(store):
    empty = store.create('manual')
    with pytest.raises(ValueError):
        store.save(empty['id'], {}, empty['revision'])
    assert len(store.catalog.foods) == 21
    assert store.catalog.data['source_sha256']
    assert all(food['ndb_number'] and len(food['per_100g']) == 4 for food in store.catalog.foods.values())
