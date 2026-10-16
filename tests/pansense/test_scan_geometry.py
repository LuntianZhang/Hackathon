import copy
import math

import cv2
import numpy as np
from PIL import Image
import pytest

from pansense import scan_geometry as geometry


def observation(azimuth,elevation,distance=.65):
    a,e=np.radians([azimuth,elevation])
    centre=distance*np.array([math.cos(e)*math.sin(a),-math.cos(e)*math.cos(a),math.sin(e)])
    forward=-centre/np.linalg.norm(centre)
    right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    rotation=np.stack([right,down,forward]);r=cv2.Rodrigues(rotation)[0];t=-rotation@centre
    k=np.array([[1050.,0,640],[0,1050.,480],[0,0,1]])
    objects=np.concatenate([geometry.marker_points(i) for i in range(len(geometry.MARKERS))])
    pixels=cv2.projectPoints(objects,r,t,k,np.zeros(5))[0].reshape(-1,2)
    return {'width':1280,'height':960,'quality':'ok','sharpness':300.,
        'marker_ids':list(range(len(geometry.MARKERS))),'objects':objects.tolist(),'pixels':pixels.tolist()},centre,k


def test_calibration_and_pose_recover_real_camera_translation_from_board():
    observations=[observation(a,e)[0] for a,e in [(0,25),(15,35),(30,50),(45,60),(60,30),(75,45),(90,55),(110,25),(140,60),(180,35)]]
    calibration=geometry.calibrate(observations)
    assert calibration is not None
    assert calibration['reprojection_error']<.01
    assert calibration['K'][0][0]==pytest.approx(1050,rel=.01)
    obs,centre,_=observation(47,32)
    pose=geometry.estimate_pose(obs,calibration)
    assert pose is not None
    np.testing.assert_allclose(pose['centre'],centre,atol=.002)
    assert pose['azimuth']==pytest.approx(47,abs=.1)
    assert pose['elevation']==pytest.approx(32,abs=.1)


def test_repeated_or_bad_frames_never_fill_a_direction_without_evidence():
    obs,centre,k=observation(0,30)
    calibration={'K':k.tolist(),'distortion':[0]*5,'width':1280,'height':960,'views':10}
    frames=[{'id':str(i),'observation':copy.deepcopy(obs),'pose':geometry.estimate_pose(obs,calibration)} for i in range(10)]
    report=geometry.coverage(frames,calibration)
    assert report['covered']==0 and report['cells'][0]['valid_frames']==1
    obs2,_,_=observation(7,30)
    frames.append({'id':'new','observation':obs2,'pose':geometry.estimate_pose(obs2,calibration)})
    report=geometry.coverage(frames,calibration)
    assert report['covered']==1 and report['cells'][0]['status']=='covered'
    frames.pop();assert geometry.coverage(frames,calibration)['covered']==0
    frames=[{'id':'unlocated','observation':obs,'pose':None}]
    assert geometry.coverage(frames,None)['pending_frames']==['unlocated']
    assert geometry.calibrate([obs]*20) is None


def test_actual_marker_detection_and_resolution_change():
    image=np.full((900,1300,3),220,np.uint8)
    dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    for identifier in range(4):
        x=80+identifier*290;y=80
        marker=cv2.aruco.generateImageMarker(dictionary,identifier,150)
        image[y:y+150,x:x+150]=marker[:,:,None]
    rng=np.random.default_rng(4);image[300:650,350:950]=rng.integers(40,220,(350,600,3),dtype=np.uint8)
    obs=geometry.image_observation(Image.fromarray(image))
    assert set(obs['marker_ids'])==set(range(4)) and obs['quality']=='ok'
    assert geometry.estimate_pose(obs,{'width':640,'height':480}) is None
    assert 'width="420mm"' in geometry.board_svg()
