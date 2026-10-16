from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import pytest

from pansense.api import create_app
from pansense.scan_api import capture_app


def picture(seed=1):
    image=Image.fromarray(np.random.default_rng(seed).integers(20,235,(240,320,3),dtype=np.uint8))
    stream=BytesIO();image.save(stream,format='JPEG');return stream.getvalue()


@pytest.fixture
def clients(tmp_path):
    with TestClient(create_app(Path.cwd(),tmp_path,jobs_enabled=False)) as desktop:
        scans=desktop.app.state.scans
        scans.public_url='https://phone.test:8443'
        with TestClient(capture_app(scans,['testserver'])) as phone:
            yield desktop,phone,scans


def create(desktop):
    response=desktop.post('/api/scans',json={'title':'手机拍摄测试'})
    assert response.status_code==201,response.text
    value=response.json();scan=value['scan'];token=value['pairing']['url'].split('/')[-1]
    return scan,{'Authorization':'Bearer '+token}


def set_status(client,scan,status,headers=None):
    latest=client.get('/api/scans/'+scan['id'],headers=headers).json()
    response=client.patch('/api/scans/'+scan['id'],headers=headers,json={'revision':latest['revision'],'status':status})
    assert response.status_code==200,response.text
    return response.json()


def upload(phone,scan,headers,index,data=None):
    return phone.post(f"/api/scans/{scan['id']}/frames",headers=headers,
        data={'frame_id':f'frame-{index}','timestamp':index},files={'file':('frame.jpg',data or picture(index),'image/jpeg')})


def test_gateway_is_scoped_and_repair_revokes_previous_phone(clients):
    desktop,phone,scans=clients
    scan,headers=create(desktop)
    assert phone.get('/api/scans').status_code in {404,405}
    assert phone.get('/api/meals').status_code==404
    assert phone.get('/api/profile').status_code==404
    assert phone.post('/api/scans',json={}).status_code in {404,405}
    assert phone.get('/api/scans/'+scan['id']).status_code==401
    assert phone.get('/api/scans/'+scan['id'],headers=headers).status_code==200
    assert phone.get('/api/scans/'+scan['id'],headers=headers).headers['cache-control']=='no-store'
    assert phone.patch('/api/scans/'+scan['id'],headers={**headers,'Origin':'https://evil.test'},json={}).status_code==403
    desktop.post('/api/scans/'+scan['id']+'/pair')
    assert phone.get('/api/scans/'+scan['id'],headers=headers).status_code==401
    assert 'token_hash' not in desktop.get('/api/scans').text
    assert phone.get('/server-key.pem').status_code==404
    static=phone.get('/capture.js?name=../../../.cache/mobile-https/ca-key.pem')
    assert static.status_code==200 and static.text.startswith('import ')


def test_saved_frames_are_idempotent_quality_checked_and_removable(clients):
    desktop,phone,scans=clients
    scan,headers=create(desktop)
    assert upload(phone,scan,headers,1).status_code==409
    scan=set_status(phone,scan,'capturing',headers)
    first=upload(phone,scan,headers,1);assert first.status_code==200,first.text
    result=first.json();assert result['frame_count']==1 and result['coverage']['covered']==0
    assert not result['frames'][0]['located'] and result['frames'][0]['quality']=='ok'
    assert upload(phone,scan,headers,1).json()['frame_count']==1
    assert upload(phone,scan,headers,1,picture(2)).status_code==409
    duplicate=upload(phone,scan,headers,2,picture(1)).json()
    assert duplicate['frames'][-1]['duplicate_view']
    assert phone.get(result['frames'][0]['url'],headers=headers).status_code==200
    assert phone.get(result['frames'][0]['url']).status_code==401
    assert upload(phone,scan,headers,3,b'not an image').status_code==400
    assert upload(phone,scan,headers,4,b'x'*(4*1024*1024+1)).status_code==413
    assert desktop.get('/api/scans').json()['scans'][0]['frame_count']==2
    latest=phone.get('/api/scans/'+scan['id'],headers=headers).json()
    assert phone.delete(f"/api/scans/{scan['id']}/frames/frame-1?revision={result['revision']}",headers=headers).status_code==409
    deleted=phone.delete(f"/api/scans/{scan['id']}/frames/frame-1?revision={latest['revision']}",headers=headers)
    assert deleted.status_code==200 and deleted.json()['frame_count']==1
    assert phone.get(result['frames'][0]['url'],headers=headers).status_code==404


def test_finish_build_and_delete_reuse_actual_scanned_photos(clients):
    desktop,phone,scans=clients
    scan,headers=create(desktop);set_status(phone,scan,'capturing',headers)
    for index in range(20): assert upload(phone,scan,headers,index).status_code==200
    assert desktop.post('/api/scans/'+scan['id']+'/build',json={'assistance':{'mode':'none'}}).status_code==400
    set_status(phone,scan,'complete',headers)
    response=desktop.post('/api/scans/'+scan['id']+'/build',json={'assistance':{'mode':'none'}})
    assert response.status_code==200,response.text
    meal=response.json();assert meal['source_kind']=='scan' and meal['analysis']['status']=='queued'
    internal=scans.store.get(meal['id']);folder=Path(internal['input_file'])
    assert len(list(folder.glob('*.jpg')))==20 and (folder/'capture.json').is_file()
    assert desktop.get(meal['photo_url']).status_code==200
    again=desktop.post('/api/scans/'+scan['id']+'/build',json={'assistance':{'mode':'none'}}).json()
    assert again['id']==meal['id']
    assert upload(phone,scan,headers,25).status_code==409
    assert desktop.delete('/api/meals/'+meal['id']).status_code==200
    assert desktop.get('/api/scans').json()['scans']==[]
    assert not (scans.root/scan['id']).exists()


def test_capture_persists_restart_and_privacy_purge_removes_images(clients):
    from pansense.scans import Scans
    desktop,phone,scans=clients
    scan,headers=create(desktop);set_status(phone,scan,'capturing',headers)
    upload(phone,scan,headers,1)
    restored=Scans(scans.store,scans.jobs,scans.project)
    assert restored.get(scan['id'])['frames'][0]['id']=='frame-1'
    assert desktop.post('/api/privacy/delete',json={'confirmation':'DELETE'}).status_code==200
    assert not (scans.root/scan['id']).exists() and scans.list()==[]


def test_manual_direction_is_labelled_not_measured(clients):
    desktop,phone,scans=clients
    scan,headers=create(desktop)
    set_status(phone,scan,'capturing',headers)
    def shot(index,sector,data=None):
        return phone.post(f"/api/scans/{scan['id']}/frames",headers=headers,
            data={'frame_id':f'manual-{index}','timestamp':index,**({} if sector is None else {'manual_sector':sector})},
            files={'file':('frame.jpg',data or picture(index),'image/jpeg')})
    assert shot(1,5).status_code==200
    flat=BytesIO();Image.new('RGB',(320,240),(128,128,128)).save(flat,format='JPEG')
    assert shot(2,6,flat.getvalue()).status_code==200
    assert shot(3,24).status_code==422 or shot(3,24).status_code==400
    value=shot(4,None).json()
    report=value['coverage']
    assert report['cells'][5]['status']=='manual'
    assert report['cells'][6]['status']=='retake'
    assert report['manual']==1 and report['covered']==0   # never counted as measured
    frames={f['id']:f for f in value['frames']}
    assert frames['manual-1']['manual'] and frames['manual-1']['sector']==5
    assert not frames['manual-4']['manual'] and frames['manual-4']['sector'] is None
    assert report['geometry_verified'] is False


def test_square_board_is_exact_size_and_detectable(clients):
    desktop,phone,scans=clients
    svg=phone.get('/api/scans/board-20cm.svg')
    assert svg.status_code==200 and 'width="200mm"' in svg.text and 'height="200mm"' in svg.text
    png=phone.get('/api/scans/board-20cm.png')
    assert png.status_code==200
    image=Image.open(BytesIO(png.content))
    assert image.size[0]==image.size[1] and abs(image.size[0]/image.info['dpi'][0]*25.4-200)<.5
