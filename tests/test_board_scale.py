from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from food3d.board_scale import SOURCE, board_hash, camera_hash, fit_board_tracks, calibrate_board
from food3d.io import write_json, read_json
from food3d.schemas import CameraFrame, save_scene
from food3d.synthetic import look_at
from food3d.volume import load_calibration, published_measurements, volume_text
from food3d.reconstruction.gaussians import file_hash
from pansense.scan_geometry import marker_points


def board_tracks(distort=False):
    random=np.random.default_rng(31);tracks={};frames=[]
    scale=3.8;translation=np.array([.04,-.02,-.5])
    for index,angle in enumerate(np.linspace(0,2*np.pi,12,endpoint=False)):
        position=translation+np.array([1.1*np.cos(angle),1.1*np.sin(angle),1.2])
        pose=look_at(position,translation)
        frame=CameraFrame(str(index),'image.jpg',960,720,[[800,0,480],[0,800,360],[0,0,1]],pose.tolist())
        frames.append(frame);projection=np.asarray(frame.K)@np.linalg.inv(pose)[:3]
        for marker in [20,22,25,27]:
            points=marker_points(marker).astype(float)
            if distort:points[:,0]*=1.4
            points=points*scale+translation
            pixels=np.column_stack([points,np.ones(4)])@projection.T
            pixels=pixels[:,:2]/pixels[:,2:]
            pixels+=random.normal(0,.12,pixels.shape)
            if index==0:pixels+=25  # corrupt camera observation, not metric data
            for corner,pixel in enumerate(pixels):
                tracks.setdefault((marker,corner),[]).append({'projection':projection,'pixel':pixel,'frame_id':frame.frame_id})
    return tracks,frames,100/scale


def test_board_recovers_centimeter_scale_with_no_paper_and_rejects_outliers():
    tracks,_,expected=board_tracks()
    result=fit_board_tracks(tracks)
    assert result['cm_per_model_unit']==pytest.approx(expected,rel=.005)
    assert result['board_type']=='square_20cm' and result['marker_size_cm']==3
    assert len(result['inlier_views'])>=10
    assert result['reprojection_rms_px']<1
    assert result['board_fit_rms_cm']<.05


def test_board_rejects_missing_geometry_and_wrong_layout():
    with pytest.raises(ValueError,match='定位板尺度校验未通过'):fit_board_tracks({})
    tracks,_,_=board_tracks(distort=True)
    with pytest.raises(ValueError,match='定位板尺度校验未通过'):fit_board_tracks(tracks)


def test_board_calibration_survives_semantics_and_cloning_but_not_camera_or_board_changes(tmp_path):
    from pansense.assistance_3d import clone_geometry
    tracks,frames,expected=board_tracks()
    bundle=tmp_path/'bundle';bundle.mkdir();(bundle/'gaussians.npz').write_bytes(b'geometry')
    write_json(bundle/'model.json',{'model_sha256':'geometry'})
    save_scene(bundle/'scene.json',frames,source='synthetic_board',world_units='test')
    (bundle/'semantics').mkdir();(bundle/'semantics/semantics.npz').write_bytes(b'semantics')
    report={'model_sha256':'geometry','semantics_sha256':file_hash(bundle/'semantics/semantics.npz'),
        'instances':[{'instance_id':1,'volume_model_units3':.01,'max_span_model_units':.4,
                      'status':'estimated','retained_points':50,'removed_outliers':0}]}
    write_json(bundle/'measurements/volumes.json',report)
    calibration={'source':SOURCE,'status':'calibrated','model_sha256':'geometry',
        'camera_sha256':camera_hash(bundle),'board_sha256':board_hash(),**fit_board_tracks(tracks)}
    write_json(bundle/'measurements/calibration.json',calibration)
    assert load_calibration(bundle,report)==calibration
    published=published_measurements(bundle)
    assert published['instances'][0]['volume_ml']==pytest.approx(.01*expected**3,rel=.015)
    assert published['board_name']=='20 × 20 cm 定位板' and published['reference_markers']==4
    assert '定位板估算' in volume_text(report,1,calibration)
    report['semantics_sha256']='changed'
    assert load_calibration(bundle,report)==calibration
    copied=tmp_path/'copy';clone_geometry(bundle,copied)
    assert load_calibration(copied,report)==calibration
    assert not (copied/'measurements/volumes.json').exists()
    calibration['board_sha256']='wrong-layout'
    write_json(bundle/'measurements/calibration.json',calibration)
    assert load_calibration(bundle,report) is None
    calibration['board_sha256']=board_hash();write_json(bundle/'measurements/calibration.json',calibration)
    frames[0].K[0][0]*=1.1
    save_scene(bundle/'scene.json',frames,source='synthetic_board',world_units='test')
    assert load_calibration(bundle,report) is None


def test_absent_board_does_not_publish_fake_metric_volume(tmp_path):
    bundle=tmp_path/'bundle';bundle.mkdir();_,frames,_=board_tracks()
    Image.new('RGB',(960,720),'white').save(bundle/'image.jpg')
    save_scene(bundle/'scene.json',frames,source='no_markers',world_units='test')
    write_json(bundle/'model.json',{'model_sha256':'geometry'})
    result=calibrate_board(bundle)
    assert result['status']=='unavailable' and 'reason' in result
    assert not (bundle/'measurements/calibration.json').exists()
    assert read_json(bundle/'measurements/board_reference.json')['detected_views']==0


def test_app_publishes_board_volume_and_reference_image_for_owned_bundle(tmp_path):
    from fastapi.testclient import TestClient
    from pansense.api import create_app
    tracks,frames,expected=board_tracks()
    with TestClient(create_app(Path.cwd(),tmp_path,jobs_enabled=False)) as client:
        store=client.app.state.store;meal=store.create('photo')
        bundle=tmp_path/'uploads'/meal['id']/'bundle';bundle.mkdir(parents=True)
        save_scene(bundle/'scene.json',frames,source='test_board',world_units='test')
        write_json(bundle/'model.json',{'model_sha256':'geometry'})
        (bundle/'semantics').mkdir();(bundle/'semantics/semantics.npz').write_bytes(b'semantics')
        calibration={'source':SOURCE,'status':'calibrated','model_sha256':'geometry',
            'camera_sha256':camera_hash(bundle),'board_sha256':board_hash(),**fit_board_tracks(tracks)}
        write_json(bundle/'measurements/calibration.json',calibration)
        Image.new('RGB',(50,40),'white').save(bundle/'measurements/board_detections.jpg')
        write_json(bundle/'measurements/volumes.json',{'model_sha256':'geometry',
            'semantics_sha256':file_hash(bundle/'semantics/semantics.npz'),
            'instances':[{'instance_id':1,'volume_model_units3':.01,'max_span_model_units':.4,'status':'estimated'}]})
        store.fields(meal['id'],bundle_ref=str(bundle),items=[store.new_item('包装',instance_id=1)])
        response=client.get(f"/api/meals/{meal['id']}")
        assert response.status_code==200,response.text
        value=response.json()
        assert value['volume_summary']['scale_source']==SOURCE
        assert value['volume_summary']['reference_markers']==4
        assert value['items'][0]['volume']['volume_ml']==pytest.approx(.01*expected**3,rel=.015)
        assert client.get(value['volume_summary']['reference_image_url']).status_code==200
