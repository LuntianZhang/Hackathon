from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from food3d.photos_only_volume import (import_photos,measure_planar_mask,shape_estimate,
    observe_photos,camera_from_board,Photo,no_pose_experiment,planar_mask_quality)
from food3d.synthetic import look_at


def test_photo_import_ignores_camera_mask_and_baseline_sidecars(tmp_path):
    source=tmp_path/'photos';source.mkdir()
    for index in range(8):Image.new('RGB',(32,24),(index,0,0)).save(source/f'{index}.jpg')
    for name in ('scene.json','transforms.json','calibration.json','volumes.json','masks.npz'):
        (source/name).write_bytes(b'not valid data, must never be read')
    photos=import_photos(source,tmp_path/'experiment')
    assert len(photos)==8 and all(p.path.parent.name=='input' for p in photos)
    assert {p.suffix for p in (tmp_path/'experiment/input').iterdir()}=={'.jpg'}
    assert all(p.path.read_bytes()==(source/f'{i}.jpg').read_bytes() for i,p in enumerate(photos))


def test_planar_measurement_and_height_ambiguity_are_explicit():
    from pansense.scan_geometry import marker_points
    points=np.concatenate([marker_points(i) for i in (20,22,25,27)])
    pixels=points[:,:2]*1000+[180,180]
    observation={'frame_id':'a','marker_ids':[20,22,25,27],'objects':points.tolist(),'pixels':pixels.tolist()}
    labels=np.zeros((360,360),np.uint16);labels[160:201,150:211]=3
    measurement=measure_planar_mask(labels,3,observation)
    assert measurement['apparent_area_cm2']==pytest.approx(24,rel=.01)
    assert measurement['apparent_short_side_cm']==pytest.approx(4,rel=.01)
    prior={'height_ratio_low':.2,'height_ratio_mid':.3,'height_ratio_high':.4,
           'fill_factor_low':.5,'fill_factor_mid':.7,'fill_factor_high':.9,'height_observable':True}
    first=shape_estimate(measurement,prior)
    assert first['volume_ml']==pytest.approx(24*4*.3*.7,rel=.01)
    # The same pixels allow a different volume when unmeasured height changes.
    prior.update(height_ratio_low=.4,height_ratio_mid=.6,height_ratio_high=.8)
    assert shape_estimate(measurement,prior)['volume_ml']==pytest.approx(2*first['volume_ml'])
    prior['height_observable']=False
    assert shape_estimate(measurement,prior)['volume_ml'] is None


def test_new_board_camera_calibration_starts_with_rendered_images(tmp_path):
    from pansense.scan_geometry import square_board_png,calibrate
    texture=cv2.imdecode(np.frombuffer(square_board_png(120),np.uint8),cv2.IMREAD_GRAYSCALE)
    side=texture.shape[0]
    source=np.float32([[0,0],[side-1,0],[side-1,side-1],[0,side-1]])
    board=np.array([[-.1,.1,0],[.1,.1,0],[.1,-.1,0],[-.1,-.1,0]])
    k=np.array([[760.,0,320],[0,760.,240],[0,0,1]])
    photos=[];truth={}
    for index,angle in enumerate(np.linspace(0,2*np.pi,18,endpoint=False)):
        center=np.array([.35*np.cos(angle),.35*np.sin(angle),.38+.1*np.cos(2*angle)])
        pose=look_at(center,np.zeros(3));inverse=np.linalg.inv(pose)
        projected=(board@inverse[:3,:3].T+inverse[:3,3])@k.T
        projected=(projected[:,:2]/projected[:,2:]).astype(np.float32)
        homography=cv2.getPerspectiveTransform(source,projected)
        image=cv2.warpPerspective(texture,homography,(640,480),borderValue=220)
        path=tmp_path/f'{index:06d}.png';Image.fromarray(image).save(path)
        photos.append(Photo(f'{index:06d}',path,640,480));truth[photos[-1].frame_id]=pose
    observations=observe_photos(photos)
    calibration=calibrate(observations)
    assert calibration is not None
    assert calibration['K'][0][0]==pytest.approx(760,rel=.04)
    valid=0
    for observation in observations:
        if len(observation['marker_ids'])<3:continue
        pose,diagnostic=camera_from_board(observation,calibration,holdout=True)
        assert np.linalg.norm(pose[:3,3]-truth[observation['frame_id']][:3,3])<.02
        assert diagnostic['heldout_marker_rms_px']<2
        valid+=1
    assert valid>=12


def test_occluded_object_without_a_reference_is_reported_without_fake_volume(tmp_path):
    evidence={'objects':[{'instance_id':3,'name':'被遮挡的下层包装'}]}
    prior={'objects':[{'instance_id':3,'reference_frame_id':'','height_observable':False,
                      'assumptions':'没有完整可见的轮廓，厚度也不可观测'}]}
    result=no_pose_experiment([],[],evidence,[],tmp_path,prior=prior)
    assert result['instances'][0]['volume_ml'] is None
    assert result['instances'][0]['status']=='height_and_complete_footprint_not_observable'


def test_ambiguous_or_tiny_mask_is_not_treated_as_complete_footprint():
    labels=np.full((20,20),65535,np.uint16);labels[:2,:2]=1
    observation={'box':[0,0,1,1]}
    assert not planar_mask_quality(labels,1,observation,np.ones_like(labels,bool))['usable_for_planar_rough_estimate']
    labels[:]=1
    assert planar_mask_quality(labels,1,observation,np.ones_like(labels,bool))['usable_for_planar_rough_estimate']
