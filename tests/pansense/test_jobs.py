from pathlib import Path
import threading

from pansense.jobs import Jobs
from pansense.nutrition import Catalog
from pansense.store import Store


def test_photo_without_segmentation_keeps_image_and_never_runs_models(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from pansense.api import create_app
    from PIL import Image
    from io import BytesIO
    import json

    with TestClient(create_app(Path.cwd(),tmp_path,jobs_enabled=False)) as client:
        stream=BytesIO();Image.new('RGB',(80,60),'orange').save(stream,format='PNG')
        response=client.post('/api/meals/upload',files={'file':('meal.png',stream.getvalue(),'image/png')},
            data={'assistance':json.dumps({'mode':'none'})})
        assert response.status_code==201,response.text
        meal=response.json();jobs=client.app.state.jobs
        def unexpected(*args,**kwargs): raise AssertionError('No model or GPU work expected')
        monkeypatch.setattr(jobs,'command',unexpected)
        monkeypatch.setattr(jobs,'release_gpu',unexpected)
        jobs.run(meal['id'],'photo',Path(client.app.state.store.get(meal['id'])['input_file']))
        ready=client.get('/api/meals/'+meal['id']).json()
        assert ready['analysis']['status']=='ready' and not ready['analysis']['error']
        assert ready['semantic_summary']=={'mode':'none','instance_count':0}
        assert not ready['items'] and ready['semantic_overlay_url'] is None
        assert client.get(ready['photo_url']).status_code==200
        assert not ready.get('three_d_analysis')

def test_cancel_waits_for_running_worker_before_assets_are_deleted(tmp_path, monkeypatch):
    import time
    store=Store(tmp_path,Catalog(Path('app/catalog/foods.json')))
    jobs=Jobs(store,Path.cwd(),threading.Lock())
    started=threading.Event()
    meal=store.create('photo')
    def pending_command(meal_id, *args, **kwargs):
        started.set()
        deadline=time.monotonic()+5
        while meal_id not in jobs.cancelled and time.monotonic()<deadline:
            time.sleep(.01)
        raise RuntimeError('分析已停止')
    monkeypatch.setattr(jobs,'command',pending_command)
    try:
        jobs.submit(meal['id'],'photo',tmp_path/'source.jpg')
        assert started.wait(3)
        jobs.cancel(meal['id'])
        assert jobs.futures[meal['id']].done()
        assert store.get(meal['id'])['analysis']['status'] != 'failed'
        assert not list((tmp_path/'uploads'/meal['id']).rglob('recognition.json'))
    finally:
        jobs.close()

def test_background_results_never_overwrite_manual_corrections_or_saved_fallback(tmp_path):
    store=Store(tmp_path,Catalog(Path('app/catalog/foods.json')))
    jobs=Jobs(store,Path.cwd(),threading.Lock(),enabled=False)
    try:
        meal=store.create('photo')
        detected=store.new_item('香蕉','banana',3,source='automatic')
        jobs.publish_items(meal['id'],[detected],tmp_path/'photo.jpg')
        meal=store.get(meal['id'])
        meal=store.edit_item(meal['id'],detected['id'],'米饭','rice',meal['revision'])
        jobs.publish_items(meal['id'],[store.new_item('苹果','apple')],tmp_path/'photo.jpg')
        assert store.get(meal['id'])['items'][0]['name']=='米饭'
        store.update_profile({'retain_media':False})
        meal=store.save(meal['id'],{},store.get(meal['id'])['revision'])
        store.fields(meal['id'],source_photo=None)
        jobs.publish_items(meal['id'],[detected],tmp_path/'late-photo.jpg')
        assert store.get(meal['id'])['source_photo'] is None
        assert store.get(meal['id'])['totals']['kcal']==195
    finally:
        jobs.close()
