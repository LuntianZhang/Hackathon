from pathlib import Path
import shutil

from fastapi.testclient import TestClient
from PIL import Image

from food3d.io import write_json
from pansense.api import create_app
from pansense.nutrition import total_nutrition


def isolated_project(tmp_path):
    project = tmp_path/'project'
    (project/'app/web').mkdir(parents=True)
    (project/'app/catalog').mkdir()
    (project/'configs').mkdir()
    for name in ('app/catalog/foods.json', 'configs/pansense.yaml'):
        shutil.copyfile(Path.cwd()/name, project/name)
    return project


def test_non_food_and_packaging_do_not_count_as_missing_or_known_nutrition():
    nutrients = {'kcal': 57, 'protein': .74, 'carbs': 14.49, 'fat': .33}
    result = total_nutrition([
        {'name': 'blueberry', 'nutrition': nutrients},
        {'name': 'unknown food', 'nutrition': None},
        {'name': 'mouse', 'object_kind': 'non_food', 'nutrition': nutrients},
        {'name': 'wrapper', 'object_kind': 'packaged_food', 'nutrition': None},
    ])
    assert result['known_items'] == result['unknown_items'] == 1
    assert result['kcal'] == 57
    assert not result['complete']
    assert total_nutrition([{'object_kind': 'non_food', 'nutrition': None}])['complete']


def test_missing_examples_are_explicit_and_do_not_create_empty_sessions(tmp_path):
    with TestClient(create_app(isolated_project(tmp_path), tmp_path/'store', jobs_enabled=False)) as client:
        status = client.get('/api/recognition/examples').json()
        assert not status['photo_available'] and status['model_url'] is None
        assert client.get('/api/recognition/reference-photo').status_code == 404
        assert client.post('/api/assist/sessions').status_code == 400
        assert client.get('/api/assist/sessions').json()['sessions'] == []


def test_reference_photo_reuses_actual_meal_model_and_preserves_binding(tmp_path):
    root = tmp_path/'store'
    with TestClient(create_app(isolated_project(tmp_path), root, jobs_enabled=False)) as client:
        store = client.app.state.store
        meal = store.create('scan', title='Existing plate')
        folder = root/'uploads'/meal['id']
        scene, bundle = folder/'scene', folder/'bundle'
        scene.mkdir(parents=True)
        bundle.mkdir()
        photo = scene/'reference.jpg'
        Image.new('RGB', (120, 80), 'orange').save(photo)
        frame = {'frame_id': 'f', 'image_path': str(photo.resolve())}
        write_json(scene/'scene.json', {'frames': [frame]})
        write_json(bundle/'scene.json', {'frames': [frame]})
        write_json(bundle/'model.json', {'source_scene': str(scene.resolve()), 'model_sha256': 'geometry'})
        (bundle/'gaussians.npz').write_bytes(b'geometry fixture')
        store.fields(meal['id'], source_photo=str(photo.resolve()), bundle_ref=str(bundle.resolve()))
        status = client.get('/api/recognition/examples').json()
        assert status['photo_available'] and status['meal_id'] == meal['id']
        assert status['model_url'] == '/#plate/' + meal['id']
        assert str(tmp_path) not in str(status)
        assert client.get('/api/recognition/reference-photo').content == photo.read_bytes()
        response = client.post('/api/assist/sessions')
        assert response.status_code == 201, response.text
        session = response.json()
        assert session['reconstruction_available']
        saved = store.get(session['id'])
        assert saved['reconstruction_binding']['bundle'] == str(bundle.resolve())
        assert saved['reconstruction_binding']['reference_frame'] == 'f'
        # Removing a model asset must remove its link, while the real photo
        # stays usable for two-dimensional recognition.
        (bundle/'gaussians.npz').unlink()
        status = client.get('/api/recognition/examples').json()
        assert status['photo_available'] and status['model_url'] is None


def test_examples_never_expose_photo_outside_owned_assets(tmp_path):
    outside = tmp_path/'private.jpg'
    Image.new('RGB', (20, 20), 'red').save(outside)
    with TestClient(create_app(isolated_project(tmp_path), tmp_path/'store', jobs_enabled=False)) as client:
        client.app.state.store.create('photo', source_photo=str(outside))
        assert not client.get('/api/recognition/examples').json()['photo_available']
