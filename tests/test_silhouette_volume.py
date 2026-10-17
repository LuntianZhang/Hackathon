import numpy as np
import pytest
from PIL import Image

from food3d.io import read_json
from food3d.schemas import load_scene
from food3d.synthetic import generate_scene, OBJECTS
from food3d.silhouette_volume import estimate_visual_hull


def test_camera_mask_only_volume_converges_on_analytic_ellipsoids(tmp_path):
    scene=tmp_path/'analytic'
    generate_scene(scene,{'views':24,'width':320,'height':240})
    _,frames=load_scene(scene/'scene.json')
    masks=[np.asarray(Image.open(scene/f.mask_path)) for f in frames]
    plane={'point':[0,0,0],'normal':[0,0,1]}
    coarse=estimate_visual_hull(frames,masks,[1,2,3],resolution=64,dilation=0,agreement=1,plane=plane)
    fine=estimate_visual_hull(frames,masks,[1,2,3],resolution=128,dilation=0,agreement=1,plane=plane)
    for obj,result in zip(OBJECTS,fine['instances']):
        truth=4*np.pi/3*np.prod(obj['radii'])
        assert result['status']=='estimated'
        assert abs(result['volume_occlusion_aware_units3']/truth-1) < .12
    for a,b in zip(coarse['instances'],fine['instances']):
        assert abs(a['volume_occlusion_aware_units3']/b['volume_occlusion_aware_units3']-1)<.08
    # A frame without a detection must not carve away the entire object.
    with_empty=estimate_visual_hull(frames+[frames[0]],masks+[np.zeros_like(masks[0])],
        [1,2,3],resolution=64,dilation=0,agreement=1,plane=plane)
    for a,b in zip(coarse['instances'],with_empty['instances']):
        assert a['retained_voxels'] == b['retained_voxels']


def test_invalid_view_or_mask_inputs_are_rejected():
    with pytest.raises(ValueError): estimate_visual_hull([],[],[1])


def test_support_plane_uses_calibrated_marker_tracks_without_metric_size(tmp_path,monkeypatch):
    import cv2
    from food3d.silhouette_volume import fiducial_support_plane
    scene=tmp_path/'board';generate_scene(scene,{'views':12,'width':160,'height':120})
    _,frames=load_scene(scene/'scene.json')
    corners=[]
    for x,y in [(-.2,-.2),(.15,-.2),(.15,.15),(-.2,.15)]:
        corners.append(np.array([[x,y,0],[x+.04,y,0],[x+.04,y+.04,0],[x,y+.04,0]]))
    class Detector:
        index=0
        def detectMarkers(self,image):
            frame=frames[self.index];index=self.index;self.index+=1
            matrix=np.asarray(frame.K)@np.linalg.inv(frame.camera_to_world)[:3]
            projections=[]
            for quad in corners:
                p=np.column_stack([quad,np.ones(4)])@matrix.T
                xy=p[:,:2]/p[:,2:]
                if index==0:xy=xy+25  # one corrupt view must not rotate the plane
                projections.append(xy.astype(np.float32)[None])
            return projections,np.arange(4),[]  # OpenCV can return flat IDs
    monkeypatch.setattr(cv2.aruco,'ArucoDetector',lambda dictionary:Detector())
    plane=fiducial_support_plane(frames,scene)
    assert plane['normal'][2]>.9999 and abs(plane['point'][2])<1e-5
    assert plane['markers']==4 and plane['corners']==16
