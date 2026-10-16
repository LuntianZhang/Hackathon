from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from food3d.io import write_json
from food3d.schemas import CameraFrame,save_scene
from pansense.api import create_app


@pytest.mark.parametrize('prepare',[False,True])
def test_scans_use_photo_colmap_pipeline_without_video_conversion(tmp_path,monkeypatch,prepare):
    with TestClient(create_app(Path.cwd(),tmp_path,jobs_enabled=False)) as client:
        store,jobs=client.app.state.store,client.app.state.jobs
        root=tmp_path/'uploads'/'scan-source';root.mkdir(parents=True)
        photo=root/'000000.jpg';Image.new('RGB',(320,240),'green').save(photo)
        meal=store.create('scan',input_file=str(root),scan_id='test-scan',assistance={'mode':'none'})
        work=tmp_path/'uploads'/meal['id']/'pipeline';work.mkdir(parents=True)
        calls=[];verified=[]
        def command(identifier,argv,*args,**kwargs):
            calls.append(argv)
            if 'photos' in argv:
                scene=Path(argv[argv.index('--scene')+1]);scene.mkdir()
                frame=CameraFrame('000000',str(photo),320,240,[[250,0,160],[0,250,120],[0,0,1]],np.eye(4).tolist())
                save_scene(scene/'scene.json',[frame],source='test',world_units='arbitrary_sfm')
            elif 'food3d.reconstruction.train_worker' in argv:
                training=Path(argv[argv.index('--output')+1]);training.mkdir();(training/'model.ckpt').touch()
            elif 'export' in argv:
                Path(argv[argv.index('--output')+1]).mkdir()
            elif 'colmap' not in argv:
                raise AssertionError(argv)
        monkeypatch.setattr(jobs,'command',command)
        monkeypatch.setattr(jobs,'verify_scan',lambda *args:verified.append(args))
        monkeypatch.setattr('pansense.jobs.subprocess.run',lambda *a,**k:pytest.fail('Photo scans must never probe/encode video'))
        jobs.video(meal['id'],root,work,work/'analysis.log',from_photos=True,prepare_only=prepare)
        done=store.get(meal['id'])
        assert done['analysis']['status']=='ready' and len(verified)==1
        assert verified[0][0]=='test-scan'
        assert [c[3] for c in calls][:2]==['photos','colmap']
        if prepare:
            assert len(calls)==2 and done['prepared_video']['reference_frame']=='000000'
            public=client.get('/api/meals/'+meal['id']).json()
            assert public['video_reference_ready']
            response=client.post('/api/meals/'+meal['id']+'/video-start',json={
                'revision':done['revision'],'assistance':{'mode':'manual','instances':[{'name':'苹果','point':[.5,.5]}]}})
            assert response.status_code==200,response.text
            assert response.json()['assistance']['mode']=='manual'
        else:
            assert done['bundle_ref'] and len(calls)==4 and done['semantic_summary']['mode']=='none'
