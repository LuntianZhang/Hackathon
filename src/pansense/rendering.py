"""Opaque frame tokens bind a displayed RGB frame to its 3D semantic pixels."""
from collections import OrderedDict
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from uuid import uuid4
import gc

import numpy as np
from PIL import Image

from food3d.io import read_json
from food3d.schemas import CameraFrame, UNKNOWN_ID, load_scene
from food3d.viewer.app import describe_click
from food3d.viewer.orbit import drag_camera
from food3d.viewer.renderer import GaussianRenderer, resize_camera

class RenderBusy(RuntimeError):
    pass

class Views:
    def __init__(self, gpu_lock):
        self.gpu_lock = gpu_lock
        self.renderer = None
        self.bundle = None
        self.frames = OrderedDict()

    def release_gpu(self):
        self.renderer = None
        self.bundle = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

    def clear_meal(self, meal_id):
        for token in [token for token, frame in self.frames.items() if frame['meal_id'] == meal_id]:
            self.frames.pop(token, None)

    def entry(self, token, meal_id):
        entry = self.frames.get(token)
        if entry is None or entry['meal_id'] != meal_id:
            raise KeyError('该视角已过期，请重新打开三维餐盘')
        return entry

    def render(self, meal, frame_id=None, base_token=None, yaw=0, pitch=0, zoom=1, food_id=None):
        if not meal.get('bundle_ref'):
            raise ValueError('这餐暂时没有可用三维结果，可使用原图和食物清单')
        if not self.gpu_lock.acquire(blocking=False):
            raise RenderBusy('正在分析另一餐，三维画面稍后可用；可以先查看原图和食物清单')
        try:
            bundle = Path(meal['bundle_ref'])
            if self.bundle != bundle:
                self.release_gpu()
                self.renderer, self.bundle = GaussianRenderer(bundle), bundle
            renderer = self.renderer
            _, cameras = load_scene(bundle/'scene.json')
            valid = (renderer.semantic_ids > 0) & (renderer.semantic_ids != UNKNOWN_ID) if renderer.semantic_ids is not None else np.zeros(len(renderer.model.means),dtype=bool)
            pivot = np.median(renderer.model.means[valid] if np.any(valid) else renderer.model.means, axis=0)
            if base_token:
                previous = self.entry(base_token, meal['id'])
                base = previous['base_camera']
                camera = drag_camera(base, pivot, yaw, pitch, zoom)
            else:
                instances = bundle/'semantics/instances.json'
                reference = read_json(instances).get('reference_frame') if instances.is_file() else None
                base = resize_camera(next((cam for cam in cameras if cam.frame_id == (frame_id or reference)), cameras[0]), 640)
                camera = base
            selected = next((item['instance_id'] for item in meal['items'] if item['id'] == food_id), 0) or 0
            result = renderer.render(camera, selected)
            stream = BytesIO()
            Image.fromarray(result['rgb']).save(stream, format='JPEG', quality=90)
            token = uuid4().hex
            self.frames[token] = {'meal_id':meal['id'], 'image':stream.getvalue(),
                  'labels':result['instance_ids'], 'confidence':result['confidence'],
                  'camera':camera, 'base_camera':base}
            # Bound CPU frame memory; several responses can coexist safely.
            while len(self.frames) > 24: self.frames.popitem(last=False)
            return {'token':token, 'image_url':f'/api/meals/{meal["id"]}/views/{token}/image',
                    'width':camera.width,'height':camera.height,'food_id':food_id,
                    'controls':{'yaw':yaw,'pitch':pitch,'zoom':zoom},
                    'captured_frames':[cam.frame_id for cam in cameras], 'frame_id':base.frame_id}
        finally:
            self.gpu_lock.release()

    def pick(self, meal, token, x, y):
        entry = self.entry(token, meal['id'])
        if meal.get('assistance',{}).get('mode') == 'none':
            return {'food_id':None,'message':'本餐未使用语义分割，请在食物清单中查看或补充食物'}
        camera = entry['camera']
        px = min(camera.width-1, int(x*camera.width))
        py = min(camera.height-1, int(y*camera.height))
        names = {item['instance_id']:item['name'] for item in meal['items'] if item.get('instance_id')}
        instance, text = describe_click(entry['labels'], entry['confidence'], px, py, names)
        item = next((item for item in meal['items'] if item.get('instance_id') == instance), None)
        if instance > 0 and item is None:
            return {'food_id':None,'message':'该食物区域已从这餐清单中移除'}
        return {'food_id':item['id'] if item else None,'message':text}
