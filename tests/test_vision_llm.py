from pathlib import Path
import copy

import numpy as np
import pytest
from PIL import Image

from food3d.io import write_json, read_json
from food3d.segmentation.hints import validate_assistance
from food3d.segmentation.vision_llm import validate_objects, select_keyframes, VisionSettings
from food3d.segmentation.llm_segment import label_scores
from food3d.schemas import CameraFrame, UNKNOWN_ID
from pansense.assistance_3d import reuse_multiview_segmentation


def detection():
    return {'objects': [{'instance_id': 3, 'name': '包装饼干', 'label': 'biscuit package',
        'kind': 'packaged_food', 'observations': [
            {'frame_id': 'a', 'box': [.1, .2, .6, .8], 'confidence': .9},
            {'frame_id': 'b', 'box': [.3, .2, .8, .7], 'confidence': .8}]}]}


def test_joint_model_ids_and_nonfood_kind_are_preserved():
    options = validate_assistance({'mode': 'llm'})
    assert options == {'mode': 'llm', 'foods': [], 'instances': []}
    value = validate_objects(detection(), ['a', 'b'])
    assert value[0]['instance_id'] == 3 and len(value[0]['observations']) == 2
    with pytest.raises(ValueError): validate_assistance({'mode': 'llm', 'food_text': 'apple'})
    with pytest.raises(ValueError): validate_assistance({'mode': 'llm', 'instances': [{'name': 'x', 'point': [.5,.5]}]})


@pytest.mark.parametrize('fault', ['bad_frame', 'pixel_coordinates', 'nan', 'duplicate_id', 'duplicate_view'])
def test_model_output_cannot_introduce_invalid_frames_boxes_or_identities(fault):
    data = detection(); obj = data['objects'][0]; observation = obj['observations'][0]
    if fault == 'bad_frame': observation['frame_id'] = '../another-scene'
    if fault == 'pixel_coordinates': observation['box'] = [1, 20, 60, 80]
    if fault == 'nan': observation['confidence'] = float('nan')
    if fault == 'duplicate_id': data['objects'].append(copy.deepcopy(obj))
    if fault == 'duplicate_view': obj['observations'].append(copy.deepcopy(observation))
    with pytest.raises(ValueError): validate_objects(data, ['a', 'b'])


def test_keyframes_keep_reference_and_cover_camera_directions():
    from food3d.synthetic import look_at
    frames = []
    for i, angle in enumerate(np.linspace(0, 2*np.pi, 12, endpoint=False)):
        pose = look_at(np.array([np.cos(angle), np.sin(angle), .6]), np.zeros(3))
        frames.append(CameraFrame(str(i), '', 10, 10, np.eye(3).tolist(), pose.tolist()))
    selected = select_keyframes(frames, '5', 4)
    assert selected[0] == 5 and len(set(selected)) == 4
    assert 11 in selected


def test_overlapping_anchor_masks_are_unknown_instead_of_arbitrary_object():
    scores = np.array([[[10., -10., 10.]], [[-10., -10., 10.]]])
    labels, confidence = label_scores(scores, [3, 7])
    assert labels.tolist() == [[3, 0, UNKNOWN_ID]]
    assert confidence[0, 2] == 0


def test_reused_multiview_result_keeps_ids_and_does_not_mutate_source(tmp_path):
    owner = tmp_path/'meal'; source = owner/'photo/multiview'; source.mkdir(parents=True)
    result = owner/'photo/recognition.json'
    write_json(result, {'segmentation_path': str(source)})
    write_json(source/'segmentation.json', {'source_scene': str(tmp_path/'scene'), 'instance_ids': [3,7],
        'masks': {'a':'masks/a.png'}, 'confidence': {'a':'confidence/a.npy'}})
    write_json(source/'instances.json', {'instances': [{'instance_id':0,'name':'背景'},
        {'instance_id':3,'name':'包装饼干'}, {'instance_id':7,'name':'鼠标'}]})
    (source/'masks').mkdir(); (source/'confidence').mkdir()
    Image.fromarray(np.array([[0,3,7]],np.uint16)).save(source/'masks/a.png')
    np.save(source/'confidence/a.npy',np.ones((1,3),np.float16))
    destination = tmp_path/'copy'
    reuse_multiview_segmentation({'id':'meal','recognition_result':str(result)},
        {'scene':str(tmp_path/'scene')},tmp_path,destination,[{'instance_id':3,'name':'已核对名称'}])
    assert np.asarray(Image.open(source/'masks/a.png')).tolist() == [[0,3,7]]
    assert np.asarray(Image.open(destination/'masks/a.png')).tolist() == [[0,3,UNKNOWN_ID]]
    assert read_json(destination/'instances.json')['instances'][1]['name'] == '已核对名称'
    assert np.load(destination/'confidence/a.npy')[0,2] == 0


def test_api_accepts_fourth_mode_and_lists_model_status(tmp_path):
    from fastapi.testclient import TestClient
    from pansense.api import create_app
    from io import BytesIO
    with TestClient(create_app(Path.cwd(),tmp_path,jobs_enabled=False)) as client:
        stream=BytesIO(); Image.new('RGB',(50,40),'red').save(stream,format='PNG')
        parent=client.post('/api/assist/sessions',files={'file':('a.png',stream.getvalue(),'image/png')}).json()
        response=client.post(f"/api/assist/sessions/{parent['id']}/runs",json={'mode':'llm'})
        assert response.status_code == 201 and response.json()['assistance']['mode'] == 'llm'
        assert 'configured' in client.get('/api/recognition/model-status').json()


def test_agent_uses_attached_images_readonly_and_structured_output(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import food3d.segmentation.vision_llm as vision
    monkeypatch.setattr(vision,'ProjectPaths',lambda:SimpleNamespace(data_root=tmp_path))
    seen={}
    def run(argv,**kwargs):
        seen['argv']=argv
        assert argv[argv.index('--sandbox')+1]=='read-only'
        assert argv.count('--image')==2 and kwargs['input'].find('a, b')>=0
        assert '只识别放在白色定位板范围内的实物' in kwargs['input']
        assert '--dangerously-bypass-approvals-and-sandbox' not in argv
        Path(argv[argv.index('-o')+1]).write_text(json.dumps(detection()),encoding='utf-8')
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr(vision.subprocess,'run',run)
    settings=VisionSettings('', 'current-agent',provider='codex',executable='codex.exe')
    request_root=tmp_path/'uploads/meal/agent_requests'
    result=vision.identify_views([('a',Image.new('RGB',(50,40))),('b',Image.new('RGB',(50,40)))],settings,
                                 request_root=request_root)
    assert result['provider']=='codex' and result['objects'][0]['instance_id']==3
    assert '--model' not in seen['argv']
    assert not list(request_root.iterdir())


def test_nonfood_and_packages_never_get_edible_catalog_nutrition(tmp_path):
    from pansense.store import Store
    from pansense.nutrition import Catalog
    store=Store(tmp_path,Catalog(Path('app/catalog/foods.json')))
    for kind in ('packaged_food','non_food'):
        item=store.new_item('香蕉',catalog_key='banana',object_kind=kind)
        assert item['catalog_key'] is None and item['nutrition'] is None


def test_agent_request_directories_are_writable_isolated_and_cleaned_on_error(tmp_path):
    from food3d.segmentation.vision_llm import agent_request_directory
    root = tmp_path/'requests'
    root.mkdir()
    sibling = root/'keep.txt'
    sibling.write_text('preserve', encoding='utf-8')
    with pytest.raises(RuntimeError, match='worker failed'):
        with agent_request_directory(root) as first:
            (first/'schema.json').write_text('{}', encoding='utf-8')
            with agent_request_directory(root) as second:
                assert first != second and first.parent == second.parent == root
                (second/'answer.json').write_text('{}', encoding='utf-8')
            assert not second.exists() and first.is_dir()
            raise RuntimeError('worker failed')
    assert not first.exists() and sibling.read_text(encoding='utf-8') == 'preserve'
    assert list(root.iterdir()) == [sibling]


@pytest.mark.parametrize('fault', ['timeout', 'failed', 'invalid_json', 'readonly_state'])
def test_failed_agent_always_removes_images_and_request_files(tmp_path, monkeypatch, fault):
    from types import SimpleNamespace
    import food3d.segmentation.vision_llm as vision
    root = tmp_path/'requests'
    def run(argv, **kwargs):
        folder = Path(argv[argv.index('--cd')+1])
        assert (folder/'schema.json').is_file() and (folder/'00.jpg').is_file()
        if fault == 'timeout': raise vision.subprocess.TimeoutExpired(argv, 1)
        if fault == 'invalid_json':
            Path(argv[argv.index('-o')+1]).write_text('invalid', encoding='utf-8')
        return SimpleNamespace(returncode=1 if fault in {'failed', 'readonly_state'} else 0,
            stderr='attempt to write a readonly database: secret-path' if fault == 'readonly_state' else '')
    monkeypatch.setattr(vision.subprocess, 'run', run)
    settings = VisionSettings('', 'current-agent', provider='codex', executable='codex.exe')
    with pytest.raises((ValueError, RuntimeError)) as caught:
        vision.identify_views([('a', Image.new('RGB', (50,40)))], settings, request_root=root)
    if fault == 'readonly_state':
        assert '本机视觉识别服务无法访问运行目录' in str(caught.value)
        assert 'secret-path' not in str(caught.value)
    assert not list(root.iterdir())


def test_bound_llm_job_passes_all_views_and_queues_same_result_for_3d(tmp_path,monkeypatch):
    import threading
    import pansense.assistance_3d
    from pansense.jobs import Jobs
    from pansense.store import Store
    from pansense.nutrition import Catalog
    store=Store(tmp_path,Catalog(Path('app/catalog/foods.json')))
    binding={'scene':str(tmp_path/'scene'),'bundle':str(tmp_path/'bundle'),'reference_frame':'a',
             'source_sha256':'source','model_sha256':'model'}
    monkeypatch.setattr(pansense.assistance_3d,'source_binding',lambda *args:binding)
    meal=store.create('photo',assistance={'mode':'llm','foods':[],'instances':[]})
    owner=tmp_path/'uploads'/meal['id'];owner.mkdir(parents=True)
    photo=owner/'source.jpg';Image.new('RGB',(50,40),'red').save(photo)
    store.fields(meal['id'],source_photo=str(photo),input_file=str(photo))
    jobs=Jobs(store,Path.cwd(),threading.Lock(),enabled=False)
    def command(identifier,argv,log,*args,**kwargs):
        assert argv[argv.index('--scene')+1]==binding['scene']
        assert argv[argv.index('--reference-frame')+1]=='a'
        output=Path(argv[argv.index('--output')+1]);output.mkdir(parents=True)
        write_json(output/'recognition.json',{'photo':str(photo),'keyframes':['a','b','c'],
            'model':'current-agent','items':[{'name':'香蕉','catalog_key':'banana','instance_id':7,
                'mask_path':str(output/'mask.png'),'recognition_score':.9,'name_source':'vision_llm',
                'kind':'packaged_food'}]})
    monkeypatch.setattr(jobs,'command',command)
    try:
        jobs.run(meal['id'],'photo',photo)
        result=store.get(meal['id'])
        assert result['analysis']['status']=='ready'
        assert result['three_d_analysis']['status']=='queued'
        assert result['items'][0]['instance_id']==7 and result['items'][0]['name_source']=='vision_llm'
        assert result['items'][0]['nutrition'] is None
        assert result['semantic_summary']['recognition_views']==3
    finally:jobs.close()
