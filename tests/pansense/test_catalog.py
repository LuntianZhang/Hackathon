import json
import math
from pathlib import Path
import re

from fastapi.testclient import TestClient
import pytest

from food3d.food_catalog import FoodIndex
from food3d.segmentation.hints import parse_foods,display_names,candidate_names
from pansense.api import create_app
from pansense.nutrition import Catalog


@pytest.fixture
def catalog():return Catalog(Path('app/catalog/foods.json'))


def test_ingredient_library_covers_categories_and_preserves_traceability(catalog):
    assert len(catalog.foods)>=700
    counts={row['key']:row['count'] for row in catalog.categories()}
    assert counts['fruit']>=50 and counts['vegetable']>=100 and counts['meat']>=100
    assert counts['staple']>=70 and counts['dairy']>=30 and counts['soy']>=15
    assert counts['nut']>=25 and counts['drink']>=35 and counts['snack']>=50
    assert counts['dish']>=70 and counts['condiment']>=40
    for food in catalog.foods.values():
        assert re.fullmatch('[a-z][a-z0-9_]*',food['key'])
        assert re.fullmatch("[a-z][a-z0-9 '\\-]*",food['recognition_label'])
        if food['nutrient_status']=='verified':
            assert food['ndb_number'] and food['source_description']
            assert all(math.isfinite(v) and v>=0 for v in food['per_100g'].values())
        else:
            assert food['per_100g'] is None and food['ndb_number'] is None
    # These closely spaced USDA IDs refer to different organs and fats.
    for key,part in [('pork_stomach','stomach'),('pork_tongue','tongue'),('beef_tripe','tripe'),('beef_tongue','tongue')]:
        assert part in catalog.foods[key]['source_description']


@pytest.mark.parametrize('name,key',[
    ('西红柿','tomato'),('奇异果','kiwi'),('  PORK BELLY  ','pork_belly'),('chicken_breast','chicken_breast'),
    ('荔枝','lychee'),('西瓜','watermelon'),('车厘子','cherry'),('五花肉','pork_belly'),
    ('羊肉','lamb'),('韭菜','chinese_chives'),('茄子','eggplant'),('油麦菜','youmai'),
    ('火龙果','pitaya'),('生西兰花','broccoli_raw'),('西兰花（水煮）','broccoli'),
    ('牛奶','milk'),('燕麦片','rolled_oats'),('油豆腐','fried_tofu'),('巴旦木','almond'),
    ('白馒头','mantou'),('云吞','wonton'),('西红柿炒鸡蛋','tomato_eggs'),
    ('奶茶','milk_tea'),('蛋挞','egg_tart'),('鸡米花','chicken_popcorn'),
    ('米粉','rice_noodles'),('面条','noodles'),('美乃滋','mayonnaise')])
def test_catalog_aliases_work_for_nutrition_and_hint_inputs(catalog,name,key):
    assert catalog.match(name)==key
    hints=parse_foods(name,list(catalog.foods.values()))
    assert hints[0]['text']==catalog.foods[key]['recognition_label']
    names=display_names({'foods':[]},list(catalog.foods.values()))
    assert names[hints[0]['text']]==catalog.foods[key]['name']
    labels=candidate_names({'foods':hints},catalog.index.labels())
    assert len(labels)==len(set(labels))


def test_unsupported_nutrients_remain_unknown_and_dishes_never_match_by_substring(catalog):
    assert catalog.match('火龙果')=='pitaya' and catalog.estimate('pitaya') is None
    assert catalog.match('番茄炒蛋包装袋') is None and catalog.match('五花肉包装袋') is None
    assert catalog.match('鼠标') is None
    assert catalog.estimate('banana')['kcal']==pytest.approx(105.02)
    assert catalog.estimate('rice')['kcal']==195
    assert catalog.estimate('pork_belly')['kcal']==518
    assert catalog.estimate('broccoli_raw')!=catalog.estimate('broccoli')


def test_recipe_and_unspecified_food_names_do_not_inherit_ingredient_nutrients(catalog):
    for name in ['番茄炒蛋','奶茶','白馒头','水饺','面条','米粉','奶酪','饼干','食用油']:
        key=catalog.match(name)
        assert key is not None and catalog.estimate(key) is None
    assert catalog.estimate(catalog.match('牛奶'))['kcal']==61
    assert catalog.estimate(catalog.match('燕麦片'))['kcal']==379
    assert catalog.estimate(catalog.match('巴旦木'))['kcal']==579
    assert catalog.estimate('rice_noodles_dry')['kcal']>catalog.estimate('rice_noodles_cooked')['kcal']
    assert catalog.match('鸡米花')!=catalog.match('炸鸡块（裹粉）')


def test_ambiguous_aliases_fail_instead_of_picking_arbitrary_nutrients():
    with pytest.raises(ValueError,match='别名重复'):
        FoodIndex([{'key':'a','name':'食物 A','aliases':['相同名称']},
                   {'key':'b','name':'食物 B','aliases':['相同名称']}])


def test_catalog_api_search_and_known_vs_pending_nutrition(tmp_path):
    with TestClient(create_app(Path.cwd(),tmp_path,jobs_enabled=False)) as client:
        data=client.get('/api/catalog').json()
        assert data['total']>=700 and len(data['foods'])==data['total']
        assert 0<data['nutrition_available']<data['total']
        found=client.get('/api/catalog',params={'q':'五花肉','category':'meat'}).json()['foods']
        assert [f['key'] for f in found]==['pork_belly']
        assert not client.get('/api/catalog',params={'q':'五花肉','category':'fruit'}).json()['foods']
        meal=client.post('/api/meals/manual').json();url='/api/meals/'+meal['id']+'/foods'
        first=client.post(url,json={'name':'五花肉','revision':meal['revision']})
        assert first.status_code==201,first.text
        value=first.json();assert value['items'][0]['catalog_key']=='pork_belly'
        assert value['items'][0]['nutrition']['kcal']==518
        pending=client.post(url,json={'name':'火龙果','revision':value['revision']}).json()
        assert pending['items'][1]['catalog_key']=='pitaya' and pending['items'][1]['nutrition'] is None
        assert pending['totals']['kcal']==518 and pending['totals']['unknown_items']==1
        dish=client.post(url,json={'name':'西红柿炒鸡蛋','revision':pending['revision']}).json()
        assert dish['items'][2]['catalog_key']=='tomato_eggs' and dish['items'][2]['nutrition'] is None
        assert dish['totals']['kcal']==518 and dish['totals']['unknown_items']==2
        assert client.get('/api/catalog',params={'q':'西红柿炒鸡蛋','category':'dish'}).json()['foods'][0]['key']=='tomato_eggs'
