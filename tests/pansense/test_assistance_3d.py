from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import threading

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from food3d.io import write_json, read_json
from food3d.segmentation.sam2 import load_seed_masks
from pansense.assistance_3d import source_binding, clone_geometry
from pansense.api import create_app
from pansense.jobs import Jobs
from pansense.nutrition import Catalog
from pansense.store import Store


def test_selected_masks_preserve_custom_names_ids_and_foreground_on_resize(tmp_path):
    mask=np.zeros((20,40),np.uint8);mask[4:16,8:32]=255
    Image.fromarray(mask).save(tmp_path/'mask.png')
    write_json(tmp_path/'seeds.json',{'instances':[{'instance_id':9,'name':'用户指定的青提','mask_path':str(tmp_path/'mask.png')}]})
    seeds,chosen=load_seed_masks(tmp_path/'seeds.json',(80,40))
    assert seeds[0]['name']=='用户指定的青提' and seeds[0]['instance_id']==9
    region=chosen[0][0]['segmentation']
    assert region.shape==(40,80) and region[20,40] and not region[0,0]


def test_source_binding_requires_exact_reference_and_never_matches_an_unrelated_photo(tmp_path):
    base=tmp_path/'data/outputs/fruit2_view_counts_run01/n30'
    reference=base/'independent_colmap_scene/images/000018.jpg';reference.parent.mkdir(parents=True)
    Image.new('RGB',(963,539),'yellow').save(reference)
    bundle=base/'independent_bundle';bundle.mkdir()
    write_json(bundle/'model.json',{'model_sha256':'test-geometry'})
    image=Image.open(reference).convert('RGB');image.thumbnail((960,960));stream=BytesIO();image.save(stream,format='JPEG',quality=95)
    photo=tmp_path/'photo.jpg';photo.write_bytes(stream.getvalue())
    parent={'input_file':str(photo),'source_sha256':sha256(photo.read_bytes()).hexdigest()}
    assert source_binding(parent,tmp_path)['reference_frame']=='000018'
    assert source_binding({**parent,'source_sha256':'wrong'},tmp_path) is None
    Image.new('RGB',(60,40),'blue').save(photo)
    assert source_binding(parent,tmp_path) is None


def test_cloning_geometry_does_not_copy_old_semantics_and_keeps_camera_image_paths_valid(tmp_path):
    original=tmp_path/'original';original.mkdir();(original/'semantics').mkdir()
    (original/'gaussians.npz').write_bytes(b'unchanged geometry')
    write_json(original/'model.json',{'model_sha256':'unchanged'})
    write_json(original/'scene.json',{'frames':[{'image_path':'../images/reference.jpg'}]})
    target=tmp_path/'new'
    clone_geometry(original,target)
    assert (target/'gaussians.npz').read_bytes()==(original/'gaussians.npz').read_bytes()
    assert not (target/'semantics').exists()
    assert Path(read_json(target/'scene.json')['frames'][0]['image_path'])==(tmp_path/'images/reference.jpg').resolve()


def test_gpu_job_queues_only_once_and_does_not_overwrite_photo_result(tmp_path):
    store=Store(tmp_path,Catalog(Path('app/catalog/foods.json')))
    item=store.new_item('青提',instance_id=2,source='manual_prompt')
    meal=store.create('photo',items=[item],analysis={'status':'ready','stage':'done','progress':100,'error':None})
    jobs=Jobs(store,Path.cwd(),threading.Lock(),enabled=False)
    try:
        jobs.submit_3d(meal['id'],{})
        value=store.get(meal['id']);revision=value['revision']
        jobs.submit_3d(meal['id'],{})
        assert store.get(meal['id'])['revision']==revision
        assert value['analysis']['status']=='ready' and value['items'][0]['id']==item['id']
        assert value['three_d_analysis']['status']=='queued'
        jobs.recover()
        assert store.get(meal['id'])['three_d_analysis']['status']=='failed'
    finally: jobs.close()


def test_unbound_photo_choice_explains_missing_views_instead_of_showing_other_scene(tmp_path):
    with TestClient(create_app(Path.cwd(),tmp_path,jobs_enabled=False)) as client:
        stream=BytesIO();Image.new('RGB',(70,50),'blue').save(stream,format='PNG')
        parent=client.post('/api/assist/sessions',files={'file':('a.png',stream.getvalue(),'image/png')}).json()
        child=client.post(f"/api/assist/sessions/{parent['id']}/runs",json={'mode':'manual','instances':[{'name':'食物','point':[.5,.5]}]}).json()
        store=client.app.state.store
        store.fields(child['id'],analysis={'status':'ready','stage':'done','progress':100,'error':None})
        response=client.post(f"/api/assist/sessions/{parent['id']}/choice",json={'run_id':child['id']})
        assert response.status_code==200
        assert not response.json()['reconstruction_available']
        value=client.get(f"/api/meals/{child['id']}").json()
        assert not value['three_d_available'] and value['model_url'] is None
        assert '多视角' in value['three_d_reason']
        assert client.get('/model.html').status_code==200
