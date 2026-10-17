from copy import deepcopy
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from pansense.api import create_app
from pansense.nutrition import Catalog, total_nutrition
from pansense.quantities import FoodQuantities
from pansense.store import Store


def report(ml=200, status='calibrated'):
    return {'status':status, 'scale_source':'pansense_aruco_board_multiview',
            'instances':[{'instance_id':1,'status':'estimated','volume_ml':ml,'max_span_cm':10}]}


@pytest.fixture
def setup_store(tmp_path):
    catalog=Catalog(Path('app/catalog/foods.json'))
    store=Store(tmp_path,catalog)
    quantities=FoodQuantities(catalog,Path('app/catalog/densities.json'),lambda _:report())
    store.quantities=quantities
    meal=store.create('test',bundle_ref=str(tmp_path/'bundle'),
        occurred_at='2026-10-17T04:00:00+00:00',items=[store.new_item('米饭',instance_id=1)])
    return store,quantities,meal


def test_food_reference_is_automatic_and_category_fills_missing_food_density(setup_store):
    store,quantities,meal=setup_store
    item=meal['items'][0]
    assert item['volume']['volume_ml']==200
    assert item['density']['status']=='food_reference' and item['density']['value_g_ml']==.73
    assert item['mass_estimate']['weight_grams']==146
    assert item['nutrition']['kcal']==189.8 and item['reference_nutrition']['kcal']==195
    assert meal['totals']['known_items']==1 and meal['totals']['quantity_pending_items']==0
    apple=store.new_item('苹果')
    assert quantities.density(apple)['status']=='category_default'
    assert quantities.density(apple)['value_g_ml']==.68
    assert quantities.candidates('apple')==[]


def test_volume_times_selected_density_updates_weight_nutrition_and_daily_summary(setup_store):
    store,quantities,meal=setup_store
    item=meal['items'][0]
    value=store.edit_item(meal['id'],item['id'],'米饭','rice',meal['revision'],
                          density={'mode':'reference','reference_id':'rice_cooked'})
    item=value['items'][0]
    assert item['density']['value_g_ml']==.73 and item['density']['kind']=='bulk'
    assert item['mass_estimate']['weight_grams']==146
    assert item['nutrition']['reference_grams']==146
    assert item['nutrition']['kcal']==pytest.approx(189.8)
    assert item['nutrition']['portion_method']=='volume_density'
    assert item['reference_nutrition']['reference_grams']==150
    assert value['totals']['volume_density_items']==1 and value['totals']['reference_items']==0
    saved=store.save(meal['id'],{},value['revision'])
    assert store.summary('2026-10-17')['totals']['kcal']==pytest.approx(189.8)
    assert store.get(meal['id'])['revision']==saved['revision']
    assert store.get(meal['id'])['items'][0]['nutrition']==item['nutrition']


def test_custom_density_unknown_food_can_have_weight_without_nutrition_and_can_be_cleared(setup_store):
    store,_,meal=setup_store
    item=meal['items'][0]
    result=store.edit_item(meal['id'],item['id'],'未收录测试食品',None,meal['revision'],
                           density={'mode':'custom','value_g_ml':.8,'note':'独立测量参考'})
    assert result['items'][0]['mass_estimate']['weight_grams']==160
    assert result['items'][0]['nutrition'] is None
    assert result['items'][0]['density']['source']=='用户填写'
    cleared=store.edit_item(meal['id'],item['id'],'未收录测试食品',None,result['revision'],density={'mode':'clear'})
    assert cleared['items'][0]['mass_estimate']['weight_grams'] is None
    assert cleared['items'][0]['volume']['volume_ml']==200


def test_changing_name_or_catalog_invalidates_saved_density(setup_store):
    store,_,meal=setup_store;item=meal['items'][0]
    before=store.edit_item(meal['id'],item['id'],'米饭','rice',meal['revision'],density={'mode':'custom','value_g_ml':.73})
    after=store.edit_item(meal['id'],item['id'],'全脂牛奶','milk',before['revision'])
    assert after['items'][0]['density']['value_g_ml']==1.03
    assert after['items'][0]['density']['status']=='food_reference'
    assert after['items'][0]['mass_estimate']['weight_grams']==206
    assert after['items'][0]['nutrition']['density_g_ml']==1.03


def test_reference_for_other_food_is_rejected_without_mutating_meal(setup_store):
    store,_,meal=setup_store
    with pytest.raises(ValueError,match='不匹配'):
        store.edit_item(meal['id'],meal['items'][0]['id'],'米饭','rice',meal['revision'],
                        density={'mode':'reference','reference_id':'milk_whole'})
    assert store.get(meal['id'])['revision']==meal['revision']


@pytest.mark.parametrize('ml,status',[(None,'calibrated'),(0,'calibrated'),(-20,'calibrated'),
                                      (float('nan'),'calibrated'),(200,'unscaled'),(200,'stale')])
def test_density_without_reliable_metric_volume_never_becomes_weight(setup_store,ml,status):
    store,quantities,meal=setup_store
    quantities.measurement_reader=lambda _:report(ml,status)
    value=store.edit_item(meal['id'],meal['items'][0]['id'],'米饭','rice',meal['revision'],
                          density={'mode':'custom','value_g_ml':.73})
    assert value['items'][0]['mass_estimate']['weight_grams'] is None
    assert value['items'][0]['nutrition'] is None


@pytest.mark.parametrize('kind',['non_food','packaged_food'])
def test_nonfood_and_packaging_cannot_become_food_weight_or_calories(setup_store,kind):
    store,quantities,_=setup_store
    item=store.new_item('测试对象',instance_id=1,object_kind=kind)
    with pytest.raises(ValueError,match='非食物对象和包装'):
        quantities.selection(item,{'mode':'custom','value_g_ml':1})
    meal=store.create('test',bundle_ref='test_bundle',items=[item])
    assert meal['items'][0]['density']['status']=='blocked'
    assert meal['items'][0]['mass_estimate']['weight_grams'] is None
    assert meal['items'][0]['nutrition'] is None


def test_two_d_record_preserves_explicit_reference_portion_instead_of_fake_volume(setup_store):
    store,_,_=setup_store
    meal=store.create('manual')
    meal=store.add_item(meal['id'],'米饭','rice',meal['revision'],density={'mode':'reference','reference_id':'rice_cooked'})
    assert meal['items'][0]['mass_estimate']['weight_grams'] is None
    assert meal['items'][0]['nutrition']['reference_grams']==150
    assert meal['totals']['reference_items']==1 and meal['totals']['volume_density_items']==0


def test_online_nutrition_is_rescaled_using_its_own_original_portion(setup_store):
    store,quantities,_=setup_store
    item=store.new_item('测试食品',instance_id=1)
    item['nutrition']={'kcal':100,'protein':4,'fat':1,'carbs':18,'reference_grams':50,'source':'test source'}
    item['density_selection']=quantities.selection(item,{'mode':'custom','value_g_ml':.8})
    meal=quantities.enrich({'bundle_ref':'test','items':[item]})
    assert meal['items'][0]['mass_estimate']['weight_grams']==160
    assert meal['items'][0]['nutrition']['kcal']==320
    assert meal['items'][0]['nutrition']['source']=='test source'
    assert total_nutrition(meal['items'])['known_items']==1


def test_density_api_catalog_edit_and_export_are_consistent(tmp_path):
    app=create_app(Path.cwd(),tmp_path,jobs_enabled=False)
    app.state.quantities.measurement_reader=lambda _:report()
    with TestClient(app) as client:
        data=client.get('/api/catalog',params={'q':'米饭'}).json()
        rice=next(food for food in data['foods'] if food['key']=='rice')
        assert rice['density_candidates'][0]['value_g_ml']==.73
        assert rice['density_default']['status']=='food_reference'
        assert len(data['category_densities'])==11
        store=app.state.store
        # Simulated geometry report only: validates conversion, not real scene accuracy.
        meal=store.create('test',title='模拟 200 mL 密度换算测试',bundle_ref=str(tmp_path/'simulated'),
                           items=[store.new_item('米饭',instance_id=1)])
        url=f"/api/meals/{meal['id']}/foods/{meal['items'][0]['id']}"
        invalid=client.patch(url,json={'revision':meal['revision'],'name':'米饭','catalog_key':'rice',
                                       'density':{'mode':'custom','value_g_ml':0}})
        assert invalid.status_code==422
        result=client.patch(url,json={'revision':meal['revision'],'name':'米饭','catalog_key':'rice',
                                      'density':{'mode':'reference','reference_id':'rice_cooked'}})
        assert result.status_code==200,result.text
        value=result.json()
        assert '_measurements' not in value
        assert value['items'][0]['volume']['volume_ml']==200
        assert value['items'][0]['mass_estimate']['weight_grams']==146
        assert value['totals']['kcal']==pytest.approx(189.8)
        exported=client.get('/api/privacy/export').json()['meals'][0]
        assert exported['items'][0]['mass_estimate']['weight_grams']==146
        assert exported['totals']==value['totals']
        stale=client.patch(url,json={'revision':meal['revision'],'name':'米饭','density':{'mode':'clear'}})
        assert stale.status_code==409


def test_orange_category_fallback_override_and_return_to_automatic(setup_store):
    store, quantities, _=setup_store
    quantities.measurement_reader=lambda _:report(263.2)
    meal=store.create('test',bundle_ref='test_bundle',items=[store.new_item('橙子',instance_id=1)])
    item=meal['items'][0]
    assert item['density']['category']=='fruit'
    assert item['density']['status']=='category_default'
    assert item['density']['sample_count']==44
    assert item['mass_estimate']['weight_grams']==pytest.approx(178.976)
    assert item['nutrition']['kcal']==pytest.approx(84.119)
    assert item['nutrition']['density_status']=='category_default'
    changed=store.edit_item(meal['id'],item['id'],'橙子','orange',meal['revision'],
                           density={'mode':'custom','value_g_ml':.9})
    assert changed['items'][0]['mass_estimate']['weight_grams']==236.88
    assert changed['items'][0]['density']['status']=='custom'
    reset=store.edit_item(meal['id'],item['id'],'橙子','orange',changed['revision'],density={'mode':'clear'})
    assert reset['items'][0]['mass_estimate']['weight_grams']==178.976
    assert reset['items'][0]['density']['status']=='category_default'
    # Repeated reads and summaries never persist or repeatedly scale estimates.
    for _ in range(2):
        assert store.get(meal['id'])['items'][0]['nutrition']['kcal']==84.119
        assert store.get(meal['id'])['revision']==reset['revision']


def test_every_catalog_food_has_a_density_but_missing_nutrients_stay_missing(setup_store):
    store, quantities, _=setup_store
    assert set(quantities.category_defaults)=={row['key'] for row in store.catalog.categories()}
    for food in store.catalog.foods.values():
        item=store.new_item(food['name'],instance_id=1)
        density=quantities.density(item)
        assert density['value_g_ml']>0
        if not quantities.candidates(food['key']):
            assert density['category']==food['category']
            assert density['status']=='category_default'
    item=store.new_item('番茄炒蛋',instance_id=1)
    meal=quantities.enrich({'bundle_ref':'test','items':[item]})
    assert meal['items'][0]['mass_estimate']['weight_grams']>0
    assert meal['items'][0]['nutrition'] is None  # density must not invent macros
    assert quantities.density(store.new_item('未分类的新食品'))['value_g_ml'] is None
    # Historical meals with an empty catalog key still get exact-name fallback.
    legacy={'name':'蓝莓','catalog_key':None}
    assert quantities.density(legacy)['category']=='fruit'
    assert quantities.density(legacy)['value_g_ml']==.68
    assert legacy=={'name':'蓝莓','catalog_key':None}
    legacy['density_selection']=quantities.selection(legacy,{'mode':'custom','value_g_ml':.9})
    assert quantities.density(legacy)['status']=='custom'


def test_category_data_rebuilds_from_pinned_source_and_preserves_sample_provenance(tmp_path):
    from tools.build_pansense_densities import build
    from food3d.io import read_json
    from statistics import mean
    result=build(output_path=tmp_path/'category_densities.json')
    assert result==read_json(Path('app/catalog/category_densities.json'))
    for category in result['categories']:
        samples=category['samples']
        assert category['sample_count']==len(samples)
        assert category['value_g_ml']==round(mean(row['value_g_ml'] for row in samples),3)
        for sample in samples:
            assert sample.get('ndb_number') or (sample.get('biblio_id') and sample.get('source_url'))
