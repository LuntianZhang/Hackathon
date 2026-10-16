from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from pansense.api import create_app


def test_same_source_three_modes_are_isolated_and_retry_keeps_inputs(tmp_path):
    with TestClient(create_app(Path.cwd(),tmp_path,jobs_enabled=False)) as client:
        stream=BytesIO(); Image.new('RGB',(200,100),'orange').save(stream,format='PNG')
        parent=client.post('/api/assist/sessions',files={'file':('test.png',stream.getvalue(),'image/png')}).json()
        ids=[]
        for options in ({'mode':'list','food_text':'苹果、香蕉'},
                        {'mode':'grounded','food_text':'苹果、香蕉'},
                        {'mode':'manual','instances':[{'instance_id':9,'name':'自定义食物','point':[.4,.5]}]}):
            response=client.post(f"/api/assist/sessions/{parent['id']}/runs",json=options)
            assert response.status_code == 201, response.text
            value=response.json(); ids.append(value['id'])
            assert value['source_sha256'] == parent['source_sha256']
            assert value['assistance']['mode'] == options['mode']
        assert len(set(ids)) == 3
        assert len(client.get(f"/api/assist/sessions/{parent['id']}/runs").json()['runs']) == 3
        assert client.get('/api/meals?drafts=true').json()['meals'] == []
        invalid=client.post(f"/api/assist/sessions/{parent['id']}/runs",json={'mode':'manual','instances':[{'name':'苹果','point':[2,.5]}]})
        assert invalid.status_code == 400
        assert client.get('/recognition.html').status_code == 200
        manual=client.app.state.store.get(ids[2]); original=manual['assistance']
        client.app.state.store.fields(ids[2],analysis={'status':'failed','stage':'test','progress':0,'error':'test'})
        assert client.post(f'/api/meals/{ids[2]}/retry').status_code == 200
        assert client.app.state.store.get(ids[2])['assistance'] == original
        assert client.post(f"/api/assist/sessions/{parent['id']}/runs",json={'mode':'list','food_text':'苹果、香蕉'}).status_code == 400
        assert client.post(f"/api/assist/sessions/{parent['id']}/choice",json={'run_id':ids[2]}).status_code == 400
        client.app.state.store.fields(ids[2],analysis={'status':'ready','stage':'test','progress':100,'error':None})
        choice=client.post(f"/api/assist/sessions/{parent['id']}/choice",json={'run_id':ids[2]})
        assert choice.status_code == 200 and choice.json()['preferred_mode'] == 'manual'
        assert client.app.state.store.get(ids[2])['assistance'] == original


def test_upload_passes_food_hints_and_custom_names_without_fabricating_catalog_keys(tmp_path):
    with TestClient(create_app(Path.cwd(),tmp_path,jobs_enabled=False)) as client:
        stream=BytesIO(); Image.new('RGB',(50,50),'yellow').save(stream,format='PNG')
        import json
        options={'mode':'list','food_text':'香蕉、特殊点心|pastry'}
        response=client.post('/api/meals/upload',files={'file':('test.png',stream.getvalue(),'image/png')},
                            data={'assistance':json.dumps(options)})
        assert response.status_code == 201
        assert response.json()['assistance']['foods'] == [{'name':'香蕉','text':'banana'},{'name':'特殊点心','text':'pastry'}]
        assert client.app.state.store.new_item('特殊点心')['nutrition'] is None
