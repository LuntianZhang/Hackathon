from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from food3d.io import write_json
from pansense.api import create_app


def photo_bytes():
    stream=BytesIO();Image.new('RGB',(120,80),'white').save(stream,format='PNG');return stream.getvalue()


def make_result(client,root,comparison=True):
    store=client.app.state.store
    parent=store.create('assist-session') if comparison else None
    meal=store.create('photo',comparison_parent=parent['id'] if parent else None,
        assistance={'mode':'manual','foods':[],'instances':[{'instance_id':7,'name':'自定义青提','box':[.1,.1,.6,.8]}]},
        analysis={'status':'ready','stage':'done','progress':100,'error':None})
    folder=root/'uploads'/meal['id'];folder.mkdir(parents=True)
    Image.new('RGB',(120,80),'white').save(folder/'source.jpg')
    mask=np.zeros((80,120),np.uint8);mask[15:65,20:70]=255
    Image.fromarray(mask).save(folder/'mask.png')
    Image.new('RGB',(120,80),'green').save(folder/'overlay.png')
    write_json(folder/'recognition.json',{'items':[]})
    item=store.new_item('自定义青提',instance_id=7,name_source='manual_prompt',needs_review=False,
                        region={'kind':'mask','path':str(folder/'mask.png')})
    return store.fields(meal['id'],items=[item],source_photo=str(folder/'source.jpg'),input_file=str(folder/'source.jpg'),
        recognition_result=str(folder/'recognition.json'))


def test_import_recognition_into_meal_is_idempotent_and_keeps_instances_and_3d(tmp_path):
    with TestClient(create_app(Path.cwd(),tmp_path,jobs_enabled=False)) as client:
        assert client.get('/').headers['cache-control']=='no-cache'
        assert client.get('/app.js').headers['cache-control']=='no-cache'
        assert client.get('/api/recognition/results').headers['cache-control']=='no-store'
        original=make_result(client,tmp_path)
        client.app.state.store.fields(original['id'],bundle_ref='test-model')
        assert client.get('/api/meals?drafts=true').json()['meals']==[]
        for _ in range(2):
            response=client.post('/api/meals/from-recognition',json={'run_id':original['id']})
            assert response.status_code==200,response.text
            meal=response.json()
            assert meal['id']==original['id'] and meal['saved_at'] is None
            assert meal['items'][0]['id']==original['items'][0]['id']
            assert meal['items'][0]['instance_id']==7 and meal['items'][0]['name']=='自定义青提'
            assert meal['three_d_available'] and meal['model_url']
        assert len(client.get('/api/meals?drafts=true').json()['meals'])==1
        assert client.get('/api/summary?date=2026-10-15').json()['count']==0
        assert client.get(meal['semantic_overlay_url']).status_code==200
        assert client.post(f"/api/meals/{meal['id']}/photo-pick",json={'x':.4,'y':.4}).json()['food_id']==meal['items'][0]['id']
        assert client.post(f"/api/meals/{meal['id']}/photo-pick",json={'x':.95,'y':.95}).json()['food_id'] is None


def test_restart_recognition_makes_new_draft_and_never_changes_saved_meal(tmp_path):
    with TestClient(create_app(Path.cwd(),tmp_path,jobs_enabled=False)) as client:
        old=make_result(client,tmp_path,comparison=False)
        old=client.app.state.store.save(old['id'],{'title':'已保存的餐食'},old['revision'])
        options={'mode':'manual','instances':[{'instance_id':12,'name':'香蕉','point':[.5,.5]}]}
        response=client.post(f"/api/meals/{old['id']}/recognize",json={'revision':old['revision'],'assistance':options})
        assert response.status_code==200,response.text
        new=response.json()
        assert new['id']!=old['id'] and new['saved_at'] is None and new['analysis']['status']=='queued'
        assert new['assistance']['instances'][0]['instance_id']==12
        after=client.app.state.store.get(old['id'])
        assert after['revision']==old['revision'] and after['items']==old['items'] and after['title']=='已保存的餐食'
        client.app.state.store.fields(old['id'],note='新的备注')
        assert client.post(f"/api/meals/{old['id']}/recognize",json={'revision':old['revision'],'assistance':options}).status_code==409


def test_invalid_manual_region_is_rejected_before_upload_creates_any_meal(tmp_path):
    with TestClient(create_app(Path.cwd(),tmp_path,jobs_enabled=False)) as client:
        import json
        bad={'mode':'manual','instances':[{'name':'香蕉','point':[1.5,.5]}]}
        response=client.post('/api/meals/upload',files={'file':('a.png',photo_bytes(),'image/png')},data={'assistance':json.dumps(bad)})
        assert response.status_code==400
        assert client.get('/api/meals?drafts=true').json()['meals']==[]
        good={'mode':'manual','instances':[{'instance_id':4,'name':'香蕉','box':[.2,.1,.8,.9]}]}
        meal=client.post('/api/meals/upload',files={'file':('a.png',photo_bytes(),'image/png')},data={'assistance':json.dumps(good)}).json()
        assert meal['assistance']['instances'][0]['instance_id']==4 and meal['source_sha256']


def test_video_label_map_overlay_only_colors_named_food_pixels(tmp_path):
    with TestClient(create_app(Path.cwd(),tmp_path,jobs_enabled=False)) as client:
        meal=make_result(client,tmp_path,comparison=False)
        mask=np.zeros((80,120),np.uint16);mask[15:65,20:70]=7;mask[0:10]=65535
        folder=tmp_path/'uploads'/meal['id'];Image.fromarray(mask).save(folder/'labels.png')
        item=meal['items'][0];item['region']={'kind':'mask','path':str(folder/'labels.png'),'instance_id':7}
        client.app.state.store.fields(meal['id'],source_kind='video',recognition_result=None,items=[item])
        public=client.get(f"/api/meals/{meal['id']}").json()
        response=client.get(public['semantic_overlay_url'])
        assert response.status_code==200
        pixels=np.asarray(Image.open(BytesIO(response.content)))
        assert not np.array_equal(pixels[40,40],pixels[75,110])
        np.testing.assert_array_equal(pixels[5,100],pixels[75,110])


def test_privacy_save_removes_semantic_images_and_visual_metadata(tmp_path):
    with TestClient(create_app(Path.cwd(),tmp_path,jobs_enabled=False)) as client:
        meal=make_result(client,tmp_path,comparison=False)
        client.app.state.store.update_profile({'retain_media':False})
        saved=client.post(f"/api/meals/{meal['id']}/save",json={'revision':meal['revision'],'title':'只留文字',
            'meal_type':'snack','occurred_at':'2026-10-15T05:00:00Z','note':''}).json()
        assert saved['photo_url'] is None and saved['semantic_overlay_url'] is None and not saved['can_refine_semantics']
        assert saved['items'][0]['name']=='自定义青提'
