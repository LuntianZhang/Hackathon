import threading
from types import SimpleNamespace

import numpy as np
import pytest

from food3d.schemas import CameraFrame, save_scene
from pansense.rendering import Views


@pytest.mark.parametrize('semantic_ids',[None,np.zeros(3,dtype=np.uint16)])
def test_geometry_without_food_regions_can_render_and_rotate(tmp_path,semantic_ids):
    frame=CameraFrame('view','image.jpg',120,80,[[100,0,60],[0,100,40],[0,0,1]],np.eye(4).tolist())
    save_scene(tmp_path/'scene.json',[frame],source='test',world_units='arbitrary_sfm')
    views=Views(threading.Lock());views.bundle=tmp_path
    def render(camera,selected):
        assert np.isfinite(camera.camera_to_world).all()
        return {'rgb':np.zeros((camera.height,camera.width,3),dtype=np.uint8),
            'instance_ids':np.zeros((camera.height,camera.width),dtype=np.uint16),
            'confidence':np.zeros((camera.height,camera.width),dtype=np.float32)}
    views.renderer=SimpleNamespace(semantic_ids=semantic_ids,
        model=SimpleNamespace(means=np.array([[0,0,-3],[1,0,-3],[-1,0,-3]])),render=render)
    meal={'id':'test','bundle_ref':str(tmp_path),'items':[],'assistance':{'mode':'none'}}
    first=views.render(meal)
    rotated=views.render(meal,base_token=first['token'],yaw=20,pitch=10,zoom=1.2)
    assert rotated['token']!=first['token'] and rotated['frame_id']=='view'
    assert views.pick(meal,rotated['token'],.5,.5)['food_id'] is None
