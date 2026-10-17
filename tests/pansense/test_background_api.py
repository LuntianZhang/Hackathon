from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from pansense.api import create_app


@pytest.mark.parametrize('enabled',[False,True])
def test_video_background_option_persists_through_interrupt_and_retry(tmp_path,enabled):
    with TestClient(create_app(Path.cwd(),tmp_path,jobs_enabled=False)) as client:
        response=client.post('/api/meals/upload?kind=video',
            files={'file':('test.mp4',b'local video fixture','video/mp4')},
            data={'board_background':str(enabled).lower(),'prepare':'true'})
        assert response.status_code==201,response.text
        meal=response.json();assert meal['board_background']==enabled
        assert client.post('/api/meals/'+meal['id']+'/interrupt').status_code==200
        assert client.post('/api/meals/'+meal['id']+'/retry').json()['board_background']==enabled


def test_background_preview_requires_ownership_and_is_cleared_for_text_only_save(tmp_path):
    with TestClient(create_app(Path.cwd(),tmp_path,jobs_enabled=False)) as client:
        store=client.app.state.store
        meal=store.create('photo',items=[store.new_item('苹果')])
        other=store.create('photo')
        folder=tmp_path/'uploads'/other['id'];folder.mkdir(parents=True)
        preview=folder/'preview.jpg';Image.new('RGB',(40,30),'white').save(preview)
        store.fields(meal['id'],background_preview=str(preview))
        url='/api/meals/'+meal['id']
        assert client.get(url+'/background-preview').status_code==404
        own=tmp_path/'uploads'/meal['id'];own.mkdir()
        path=own/'preview.jpg';Image.new('RGB',(40,30),'white').save(path)
        store.fields(meal['id'],background_preview=str(path),background_summary={'enabled':True})
        public=client.get(url).json()
        assert 'background_preview' not in public and client.get(public['background_preview_url']).status_code==200
        store.update_profile({'retain_media':False})
        saved=client.post(url+'/save',json={'revision':public['revision'],'title':'食物',
            'meal_type':'snack','occurred_at':meal['occurred_at'],'note':''})
        assert saved.status_code==200,saved.text
        assert saved.json()['background_preview_url'] is None and not saved.json()['background_summary']
        assert client.get(url+'/background-preview').status_code==404
