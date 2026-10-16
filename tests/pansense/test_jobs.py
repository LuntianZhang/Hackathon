from pathlib import Path
import threading

from pansense.jobs import Jobs
from pansense.nutrition import Catalog
from pansense.store import Store

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
