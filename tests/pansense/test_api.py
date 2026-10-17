from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from pansense.api import create_app

@pytest.fixture
def client(tmp_path):
    # Business-route tests must not depend on ignored local reconstruction data.
    import shutil
    from food3d.io import write_json
    from food3d.schemas import CameraFrame, save_scene
    import numpy as np
    project=tmp_path/'project'
    (project/'app/catalog').mkdir(parents=True)
    shutil.copyfile('app/catalog/foods.json',project/'app/catalog/foods.json')
    shutil.copytree('app/web',project/'app/web')
    (project/'configs').mkdir()
    shutil.copyfile('configs/pansense.yaml',project/'configs/pansense.yaml')
    bundle=project/'data/outputs/fruit2_bundle'
    (bundle/'semantics').mkdir(parents=True)
    Image.new('RGB',(60,40),'white').save(bundle/'source.jpg')
    frame=CameraFrame('a','source.jpg',60,40,[[50,0,30],[0,50,20],[0,0,1]],np.eye(4).tolist())
    save_scene(bundle/'scene.json',[frame],source='test_demo_fixture',world_units='test_units')
    write_json(bundle/'semantics/instances.json',{'reference_frame':'a','instances':[
        {'instance_id':0,'name':'背景'},{'instance_id':1,'name':'苹果'},
        {'instance_id':2,'name':'香蕉'},{'instance_id':3,'name':'面包'}]})
    write_json(bundle/'semantics/fusion.json',{'segmentation':str(bundle)})
    (bundle/'masks').mkdir();Image.fromarray(np.zeros((40,60),np.uint16)).save(bundle/'masks/a.png')
    with TestClient(create_app(project, tmp_path/'store', jobs_enabled=False)) as client:
        yield client

def test_real_record_crud_and_day_summary(client):
    meal = client.post('/api/meals/manual').json()
    identifier = meal['id']
    meal = client.post(f'/api/meals/{identifier}/foods',json={'name':'香蕉','catalog_key':'banana','revision':meal['revision']}).json()
    item = meal['items'][0]
    updated = client.patch(f'/api/meals/{identifier}/foods/{item["id"]}',json={'name':'苹果','catalog_key':'apple','revision':meal['revision']}).json()
    assert updated['totals']['kcal'] == pytest.approx(94.64)
    stale = client.delete(f'/api/meals/{identifier}/foods/{item["id"]}?revision={meal["revision"]}')
    assert stale.status_code == 409
    body = {'title':'验收餐食','meal_type':'lunch','occurred_at':'2026-10-12T04:15:00.000Z','note':'测试','revision':updated['revision']}
    saved = client.post(f'/api/meals/{identifier}/save',json=body)
    assert saved.status_code == 200
    assert client.get('/api/summary?date=2026-10-12').json()['count'] == 1
    assert client.get('/api/summary?date=2026-10-11').json()['count'] == 0
    exported = client.get('/api/privacy/export').json()
    assert exported['meals'][0]['title'] == '验收餐食'
    assert client.delete(f'/api/meals/{identifier}').status_code == 200
    assert client.get(f'/api/meals/{identifier}').status_code == 404

def test_photo_validation_region_and_2d_fallback(client):
    invalid = client.post('/api/meals/upload',files={'file':('bad.jpg',b'not an image','image/jpeg')})
    assert invalid.status_code == 400
    assert client.get('/api/meals?drafts=true').json()['meals'] == []
    stream = BytesIO(); Image.new('RGB',(60,40),'orange').save(stream,format='PNG')
    meal = client.post('/api/meals/upload',files={'file':('中文照片.png',stream.getvalue(),'image/png')}).json()
    assert meal['photo_url'] and not meal['three_d_available']
    assert 'input_file' not in meal
    assert client.get(meal['photo_url']).status_code == 200
    identifier = meal['id']
    bad = client.post(f'/api/meals/{identifier}/foods',json={'name':'未知','revision':meal['revision'],'region':{'bbox':[-.1,0,1,1]}})
    assert bad.status_code == 400
    meal = client.post(f'/api/meals/{identifier}/foods',json={'name':'米饭','revision':meal['revision'],
                          'region':{'bbox':[.1,.1,.8,.8]}}).json()
    region = client.get(meal['items'][0]['region_url'])
    assert region.status_code == 200 and region.headers['content-type'] == 'image/png'
    assert client.post(f'/api/meals/{identifier}/photo-pick',json={'x':.4,'y':.4}).json()['food_id']==meal['items'][0]['id']
    assert client.post(f'/api/meals/{identifier}/photo-pick',json={'x':.95,'y':.95}).json()['food_id'] is None
    assert client.post(f'/api/meals/{identifier}/view',json={}).status_code == 400
    saved = client.post(f'/api/meals/{identifier}/save',json={'title':'二维回退','meal_type':'snack',
        'occurred_at':'2026-10-12T15:00:00+08:00','revision':meal['revision']})
    assert saved.status_code == 200

def test_privacy_media_preference_purge_and_cross_origin_protection(client):
    assert client.patch('/api/profile',json={'onboarded':True,'goal_kcal':1900,'retain_media':False}).status_code == 200
    meal = client.post('/api/meals/demo',json={'scene':'fruit2'}).json()
    saved = client.post(f'/api/meals/{meal["id"]}/save',json={'title':'仅保存营养','meal_type':'snack',
        'occurred_at':'2026-10-12T15:00:00+08:00','revision':meal['revision']}).json()
    assert saved['photo_url'] is None and not saved['three_d_available']
    assert saved['analysis']['status'] == 'ready'
    assert client.post('/api/privacy/delete',json={'confirmation':'no'}).status_code == 400
    assert client.post('/api/privacy/delete',json={'confirmation':'DELETE'}).json()['deleted']
    assert client.get('/api/profile').json()['onboarded'] is False
    assert client.get('/api/meals?drafts=true').json()['meals'] == []
    assert client.post('/api/meals/manual',headers={'origin':'https://evil.example'}).status_code == 403
    assert client.get('/api/health',headers={'host':'evil.example'}).status_code == 400
    assert client.get('/api/catalog').headers['cache-control'] == 'no-store'

def test_invalid_input_and_demo_are_not_mock_recognition(client):
    assert client.patch('/api/profile',json={'timezone':'not-a-timezone'}).status_code == 400
    assert client.post('/api/meals/demo',json={'scene':'../../'}).status_code == 400
    assert client.patch('/api/profile',content='{"goal_kcal":Infinity}',headers={'content-type':'application/json'}).status_code == 422
    meal = client.post('/api/meals/demo',json={'scene':'fruit2'}).json()
    assert meal['source_kind'] == 'demo' and meal['three_d_available']
    assert len(meal['items']) == 3
    assert {item['instance_id'] for item in meal['items']} == {1,2,3}
