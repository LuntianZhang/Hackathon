from hashlib import sha256
from pathlib import Path
import shutil

from fastapi.testclient import TestClient
from PIL import Image

from food3d.io import read_json, write_json
from pansense.api import create_app
from pansense.assistance_3d import normalized_photo, source_binding


def model_scene(project):
    scene=project/'data/scenes/own_video';(scene/'images').mkdir(parents=True)
    photo=scene/'images/view17.jpg';Image.new('RGB',(120,80),'yellow').save(photo)
    frame={'frame_id':'view17','image_path':'images/view17.jpg'}
    write_json(scene/'scene.json',{'frames':[frame]})
    bundle=project/'data/outputs/own_model';bundle.mkdir(parents=True)
    (bundle/'gaussians.npz').write_bytes(b'original geometry')
    write_json(bundle/'model.json',{'model_sha256':'own-model','source_scene':str(scene)})
    write_json(bundle/'scene.json',{'frames':[{**frame,'image_path':str(photo)}]})
    return photo,bundle


def test_binding_follows_own_model_camera_and_rejects_mismatched_photo(tmp_path):
    photo,bundle=model_scene(tmp_path)
    binding=source_binding({'bundle_ref':str(bundle),'source_photo':str(photo)},tmp_path)
    assert binding['reference_frame']=='view17' and binding['model_sha256']=='own-model'
    copy=tmp_path/'data/copy.jpg';copy.write_bytes(normalized_photo(photo))
    child={'reconstruction_binding':binding,'input_file':str(copy),'source_sha256':sha256(copy.read_bytes()).hexdigest()}
    assert source_binding(child,tmp_path)==binding
    write_json(bundle/'semantics/instances.json',{'reference_frame':'view17'})
    old_photo={'source_kind':'photo','bundle_ref':str(bundle),'input_file':str(copy),'source_sha256':child['source_sha256']}
    assert source_binding(old_photo,tmp_path)==binding
    Image.new('RGB',(120,80),'blue').save(copy)
    assert source_binding(child,tmp_path) is None
    copy.write_bytes(normalized_photo(photo))
    write_json(bundle/'model.json',{'model_sha256':'different-model','source_scene':binding['scene']})
    assert source_binding(child,tmp_path) is None


def test_demo_assistance_preserves_geometry_binding_and_auto_queues_fusion(tmp_path,monkeypatch):
    photo,bundle=model_scene(tmp_path)
    (tmp_path/'app/web').mkdir(parents=True)
    for relative in ['app/catalog/foods.json','configs/pansense.yaml']:
        target=tmp_path/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path.cwd()/relative,target)
    with TestClient(create_app(tmp_path,tmp_path/'data/app',jobs_enabled=False)) as client:
        store=client.app.state.store;jobs=client.app.state.jobs
        old=store.create('demo',source_photo=str(photo),bundle_ref=str(bundle),
            items=[store.new_item('旧名字',instance_id=9)])
        public=client.get('/api/meals/'+old['id']).json()
        assert public['can_refine_semantics'] and public['semantic_targets_3d']
        response=client.post(f"/api/meals/{old['id']}/recognize",json={'revision':old['revision'],
            'assistance':{'mode':'manual','instances':[{'instance_id':3,'name':'香蕉','point':[.5,.5]}]}})
        assert response.status_code==200,response.text
        child=response.json();assert child['semantic_auto_3d'] and child['reconstruction_available']
        assert not child['three_d_available'] and 'reconstruction_binding' not in child
        internal=store.get(child['id'])
        assert internal['reconstruction_binding']['bundle']==str(bundle)
        assert Path(internal['input_file']).read_bytes()==normalized_photo(photo)
        def recognize(meal_id,argv,log,*args,**kwargs):
            output=Path(argv[argv.index('--output')+1]);output.mkdir(parents=True)
            mask=output/'mask.png';Image.new('L',(120,80),255).save(mask)
            write_json(output/'recognition.json',{'photo':internal['input_file'],'items':[
                {'name':'香蕉','catalog_key':'banana','instance_id':3,'name_source':'manual_prompt',
                 'mask_path':str(mask),'recognition_score':1.0}]})
        monkeypatch.setattr(jobs,'command',recognize)
        monkeypatch.setattr(jobs,'release_gpu',lambda:None)
        jobs.run(child['id'],'photo',Path(internal['input_file']))
        ready=store.get(child['id'])
        assert ready['analysis']['status']=='ready'
        assert ready['three_d_analysis']['status']=='queued'
        assert ready['items'][0]['instance_id']==3 and ready['items'][0]['name']=='香蕉'
        assert store.get(old['id'])['items'][0]['name']=='旧名字'
        assert (bundle/'gaussians.npz').read_bytes()==b'original geometry'


def test_video_refinement_reads_reference_photo_instead_of_mp4(tmp_path):
    photo,bundle=model_scene(tmp_path)
    (tmp_path/'app/web').mkdir(parents=True)
    for relative in ['app/catalog/foods.json','configs/pansense.yaml']:
        target=tmp_path/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(Path.cwd()/relative,target)
    with TestClient(create_app(tmp_path,tmp_path/'data/app',jobs_enabled=False)) as client:
        store=client.app.state.store
        video=tmp_path/'data/video.mp4';video.write_bytes(b'video bytes are not a JPEG')
        meal=store.create('video',source_photo=str(photo),input_file=str(video),bundle_ref=str(bundle))
        assert source_binding(meal,tmp_path)['reference_frame']=='view17'
        response=client.post(f"/api/meals/{meal['id']}/recognize",json={'revision':meal['revision'],
            'assistance':{'mode':'manual','instances':[{'name':'物品','point':[.5,.5]}]}})
        assert response.status_code==200,response.text
        child=store.get(response.json()['id'])
        assert child['semantic_auto_3d']
        assert Path(child['input_file']).read_bytes()==normalized_photo(photo)
