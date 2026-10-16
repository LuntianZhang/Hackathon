"""Video prompts must refer to the exact registered, undistorted reference frame."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from food3d.schemas import CameraFrame, save_scene
from pansense.api import create_app


def reference_scene(folder):
    folder.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(3):
        name = f'{index:06d}'
        Image.new('RGB', (120, 80), (index*50, 100, 80)).save(folder/f'{name}.jpg')
        frames.append(CameraFrame(name, f'{name}.jpg', 120, 80,
            [[100, 0, 60], [0, 100, 40], [0, 0, 1]], np.eye(4).tolist()))
    save_scene(folder/'scene.json', frames, source='test', world_units='arbitrary_sfm')
    return frames


def test_prepare_does_not_start_recognition_and_manual_start_is_revision_checked(tmp_path, monkeypatch):
    with TestClient(create_app(Path.cwd(), tmp_path, jobs_enabled=False)) as client:
        response = client.post('/api/meals/upload?kind=video',
            files={'file': ('video.mp4', b'video test fixture', 'video/mp4')}, data={'prepare': 'true'})
        assert response.status_code == 201, response.text
        meal = response.json()
        store, jobs = client.app.state.store, client.app.state.jobs
        assert meal['video_preparing'] and not meal['video_reference_ready']
        calls = []
        def command(identifier, argv, log, *args, **kwargs):
            calls.append(argv)
            if 'colmap' in argv:
                reference_scene(Path(argv[argv.index('--scene')+1]))
        monkeypatch.setattr(jobs, 'command', command)
        monkeypatch.setattr('pansense.jobs.subprocess.run', lambda *a, **k:
            SimpleNamespace(stdout=json.dumps({'format': {'duration': '10'}})))
        work = tmp_path/'uploads'/meal['id']/'preparation'
        work.mkdir()
        jobs.video(meal['id'], work.parent/'source.mp4', work, work/'analysis.log', prepare_only=True)
        assert [c[3] for c in calls] == ['extract', 'colmap']
        ready = client.get(f"/api/meals/{meal['id']}").json()
        assert ready['video_reference_ready'] and not ready['video_preparing']
        assert 'prepared_video' not in ready  # No filesystem paths in the public contract.
        assert not ready['items'] and ready['analysis']['status'] == 'ready'
        assert client.get(ready['photo_url']).status_code == 200
        options = {'mode': 'manual', 'instances': [{'instance_id': 7, 'name': '自定义水果', 'point': [.4, .5]}]}
        body = {'revision': ready['revision'], 'assistance': options, 'paper_present': False}
        stale = {**body, 'revision': meal['revision']}
        assert client.post(f"/api/meals/{meal['id']}/video-start", json=stale).status_code == 409
        started = client.post(f"/api/meals/{meal['id']}/video-start", json=body)
        assert started.status_code == 200, started.text
        assert started.json()['assistance']['instances'][0]['instance_id'] == 7
        assert not started.json()['paper_present'] and not started.json()['video_reference_ready']
        assert client.post(f"/api/meals/{meal['id']}/video-start", json=body).status_code == 409

        # The next phase must reuse this scene/reference, never extract another seed.
        calls.clear()
        class ReachedSegmentation(Exception): pass
        def stop_at_segmentation(identifier, argv, *args, **kwargs):
            calls.append(argv)
            raise ReachedSegmentation
        monkeypatch.setattr(jobs, 'command', stop_at_segmentation)
        training = work.parent/'training-test'; training.mkdir()
        with pytest.raises(ReachedSegmentation):
            jobs.video(meal['id'], work.parent/'source.mp4', training, training/'analysis.log')
        argv = calls[0]
        assert argv[3] == 'segment'
        assert argv[argv.index('--scene')+1] == str(work/'scene')
        assert argv[argv.index('--reference-frame')+1] == '000002'


def test_video_start_rejects_missing_reference_invalid_regions_and_wrong_kind(tmp_path):
    with TestClient(create_app(Path.cwd(), tmp_path, jobs_enabled=False)) as client:
        store = client.app.state.store
        meal = store.create('video')
        url = f"/api/meals/{meal['id']}/video-start"
        body = {'revision': meal['revision'], 'assistance': {'mode': 'auto'}}
        assert client.post(url, json=body).status_code == 400
        body['assistance'] = {'mode': 'manual', 'instances': [{'name': '苹果', 'point': [2, .5]}]}
        assert client.post(url, json=body).status_code == 400
        assert client.post('/api/meals/upload', files={'file': ('a.jpg', b'x', 'image/jpeg')},
                           data={'prepare': 'true'}).status_code == 400
        assert len(store.list(include_drafts=True)) == 1


@pytest.mark.parametrize('paper_present', [False, True])
def test_prepared_video_keeps_manual_identity_and_respects_paper_choice(tmp_path, monkeypatch, paper_present):
    with TestClient(create_app(Path.cwd(), tmp_path, jobs_enabled=False)) as client:
        store, jobs = client.app.state.store, client.app.state.jobs
        meal = store.create('video', paper_present=paper_present, assistance={'mode': 'manual', 'foods': [],
            'instances': [{'instance_id': 9, 'name': '自定义水果', 'point': [.4, .5]}]})
        root = tmp_path/'uploads'/meal['id']; scene = root/'prepared-scene'
        reference_scene(scene)
        store.fields(meal['id'], prepared_video={'scene': str(scene), 'reference_frame': '000001'})
        work = root/'pipeline'; work.mkdir()
        calls = []
        def command(identifier, argv, *args, **kwargs):
            calls.append(argv)
            if 'segment' in argv:
                folder = Path(argv[argv.index('--output')+1]); folder.mkdir()
                (folder/'instances.json').write_text(json.dumps({'instances': [
                    {'instance_id': 0, 'name': '背景'}, {'instance_id': 9, 'name': '自定义水果'}]}), encoding='utf-8')
            elif 'food3d.reconstruction.train_worker' in argv:
                folder = work/'training'; folder.mkdir(); (folder/'model.ckpt').touch()
            elif 'export' in argv:
                (work/'bundle').mkdir()
            elif 'evaluate' in argv:
                folder = work/'evaluation'; folder.mkdir()
                (folder/'report.json').write_text(json.dumps({'mean_instance_iou': .9, 'food_click_accuracy': .9}))
        monkeypatch.setattr(jobs, 'command', command)
        monkeypatch.setattr('pansense.jobs.subprocess.run', lambda *a, **k:
            SimpleNamespace(stdout=json.dumps({'format': {'duration': '10'}})))
        jobs.video(meal['id'], root/'source.mp4', work, work/'analysis.log')
        done = store.get(meal['id'])
        assert done['analysis']['status'] == 'ready' and done['bundle_ref']
        assert done['items'][0]['instance_id'] == 9 and done['items'][0]['name'] == '自定义水果'
        assert done['items'][0]['source'] == 'manual_prompt' and not done['items'][0]['needs_review']
        assert done['items'][0]['region']['path'].endswith('000001.png')
        assert any('paper-scale' in c for c in calls) == paper_present
        assert not any('extract' in c or 'colmap' in c for c in calls)
