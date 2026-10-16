from pathlib import Path

import cv2
import numpy as np
import pytest

from food3d.io import write_json
from food3d.paper_scale import SOURCE,camera_hash,detect_paper,fit_scale,observation,rectangle
from food3d.schemas import CameraFrame
from food3d.volume import load_calibration,published_measurements
from food3d.reconstruction.gaussians import file_hash


def projected_observations(noise=.1):
    rng=np.random.default_rng(29);values=[];scale=20.0
    for i,angle in enumerate(np.linspace(0,2*np.pi,24,endpoint=False)):
        c=np.array([2*np.cos(angle),2*np.sin(angle),2.0])
        forward=-c/np.linalg.norm(c);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
        down=np.cross(forward,right);pose=np.eye(4);pose[:3,:3]=np.stack([right,down,forward],axis=1);pose[:3,3]=c
        k=np.array([[700.,0,319.5],[0,700,239.5],[0,0,1]])
        frame=CameraFrame(str(i),'image.jpg',640,480,k.tolist(),pose.tolist())
        points=(rectangle(10,2)/scale-c)@pose[:3,:3]
        uv=points@k.T;uv=uv[:,:2]/uv[:,2:]+rng.normal(0,noise,(4,2))
        # A plain paper's endpoints do not have unique visual IDs.
        if i%2: uv=np.roll(uv,2,axis=0)
        value=observation(frame,uv)
        assert value is not None
        values.append(value)
    return values


def test_paper_recovers_metric_scale_with_ambiguous_corners_and_bad_views():
    values=projected_observations()
    values[2]['offset_cm']+=np.array([40,20,-30]);values[15]['offset_cm']+=np.array([-30,40,20])
    result=fit_scale(values)
    assert result['cm_per_model_unit']==pytest.approx(20,rel=.02)
    assert result['short_edge_cm_check']==pytest.approx(2,rel=.03)
    assert len(result['inlier_views'])>=18 and '2' not in result['inlier_views'] and '15' not in result['inlier_views']
    assert result['reprojection_rms_px']<1


def test_white_rectangle_detection_ignores_round_objects_and_clipped_paper():
    image=np.zeros((400,700,3),np.uint8)+[110,65,35];image=image.astype(np.uint8)
    cv2.fillConvexPoly(image,np.array([[100,80],[480,120],[472,185],[92,145]]),(235,235,235))
    cv2.putText(image,'10 cm x 2 cm',(180,135),cv2.FONT_HERSHEY_SIMPLEX,.5,(70,70,70),1)
    cv2.circle(image,(560,300),45,(245,245,245),-1)
    cv2.rectangle(image,(0,250),(220,280),(255,255,255),-1)
    detections=detect_paper(image)
    assert len(detections)==1 and detections[0]['corners'].shape==(4,2)


def test_missing_views_static_camera_and_invalid_dimensions_are_rejected():
    values=projected_observations(0)
    with pytest.raises(ValueError,match='不足'): fit_scale(values[:3])
    with pytest.raises(ValueError): fit_scale([values[0]]*10)
    with pytest.raises(ValueError): rectangle(0,2)


def test_paper_scale_survives_new_semantics_but_not_geometry_or_camera_changes(tmp_path):
    bundle=tmp_path/'bundle';bundle.mkdir()
    frame=projected_observations(0)[0]['frame']
    from dataclasses import asdict
    write_json(bundle/'scene.json',{'frames':[asdict(frame)]})
    write_json(bundle/'model.json',{'model_sha256':'same-geometry'})
    (bundle/'semantics').mkdir();(bundle/'semantics/semantics.npz').write_bytes(b'semantic-version-2')
    report={'model_sha256':'same-geometry','semantics_sha256':file_hash(bundle/'semantics/semantics.npz'),
        'instances':[{'instance_id':1,'volume_model_units3':.125,'max_span_model_units':1.0,'status':'estimated'}]}
    write_json(bundle/'measurements/volumes.json',report)
    calibration={'source':SOURCE,'status':'calibrated','model_sha256':'same-geometry',
        'camera_sha256':camera_hash(bundle),'cm_per_model_unit':20.,'inlier_views':['a','b','c','d','e','f'],
        'paper_length_cm':10,'paper_width_cm':2}
    write_json(bundle/'measurements/calibration.json',calibration)
    assert load_calibration(bundle,report)==calibration
    assert published_measurements(bundle)['instances'][0]['volume_ml']==pytest.approx(1000)
    # A new semantic version changes volumes, not the metric coordinate scale.
    report['semantics_sha256']='new-version'
    assert load_calibration(bundle,report)==calibration
    assert published_measurements(bundle)['status']=='calibrated'
    frame.K[0][0]*=1.1
    write_json(bundle/'scene.json',{'frames':[asdict(frame)]})
    assert load_calibration(bundle,report) is None
    assert published_measurements(bundle)['status']=='unscaled'


def test_semantic_geometry_copy_preserves_paper_scale_without_old_volumes(tmp_path):
    from dataclasses import asdict
    from pansense.assistance_3d import clone_geometry
    source=tmp_path/'source';source.mkdir();(source/'gaussians.npz').write_bytes(b'geometry')
    write_json(source/'model.json',{'model_sha256':'same'})
    frame=projected_observations()[0]['frame']
    write_json(source/'scene.json',{'frames':[asdict(frame)]})
    calibration={'source':SOURCE,'model_sha256':'same','camera_sha256':camera_hash(source),
                 'status':'calibrated','cm_per_model_unit':20.}
    write_json(source/'measurements/calibration.json',calibration)
    write_json(source/'measurements/volumes.json',{'must_not_copy':'old semantics'})
    target=tmp_path/'copy';clone_geometry(source,target)
    assert not (target/'measurements/volumes.json').exists()
    assert load_calibration(target,{'model_sha256':'same'})==calibration
