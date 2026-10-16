"""Compare the adapter against the installed Nerfstudio camera and renderer."""
from pathlib import Path
from food3d.paths import ProjectPaths, resolve_arguments
import argparse
import os
os.environ['TORCH_HOME'] = str(ProjectPaths().cache_root/'models/torch')
import numpy as np
import torch
from PIL import Image
from nerfstudio.utils.eval_utils import eval_setup
from food3d.schemas import load_scene
from food3d.viewer.renderer import GaussianRenderer, resize_camera
from food3d.io import read_json, write_json

p=argparse.ArgumentParser()
p.add_argument('--training',type=Path,default=ProjectPaths().resolve('data/outputs/fruit1_refined_training/latest_training.json'))
p.add_argument('--bundle',type=Path,default=ProjectPaths().resolve('data/outputs/fruit1_final'))
p.add_argument('--frame',default='000077')
p.add_argument('--output',type=Path,default=ProjectPaths().resolve('data/outputs/native_parity'))
args=resolve_arguments(p.parse_args())
args.output.mkdir(parents=True,exist_ok=True)
run = Path(read_json(args.training)['config']).parent
config, pipeline, checkpoint, step = eval_setup(run/'config.yml', test_mode='inference')
_, frames = load_scene(args.bundle/'scene.json')
frame=next(f for f in frames if f.frame_id==args.frame)
dataset = pipeline.datamanager.train_dataset
paths = dataset.image_filenames
index = next(i for i,p in enumerate(paths) if p.stem == args.frame)
native = dataset.cameras[index:index+1].to('cuda')
expected = np.asarray(frame.camera_to_world) @ np.diag([1,-1,-1,1])
print('native', native.camera_to_worlds.cpu().numpy(), 'adapter', expected, 'K', native.get_intrinsics_matrices())
print('pose_error', np.max(np.abs(native.camera_to_worlds[0].cpu().numpy()-expected[:3])))
with torch.inference_mode():
    outputs = pipeline.model.get_outputs(native)
native_rgb=outputs['rgb'].clamp(0,1).cpu().numpy()
Image.fromarray((native_rgb*255).astype('uint8')).save(args.output/'native.png')
renderer = GaussianRenderer(args.bundle)
rendered,_,_=renderer.render_geometry(frame)
adapter_rgb=rendered[0,:,:,:3].clamp(0,1).cpu().numpy()
Image.fromarray((adapter_rgb*255).astype('uint8')).save(args.output/'adapter.png')
result={'frame_id':args.frame,'max_pose_error':float(np.max(np.abs(native.camera_to_worlds[0].cpu().numpy()-expected[:3]))),'max_rgb_error':float(np.max(np.abs(native_rgb-adapter_rgb))),'mean_rgb_error':float(np.mean(np.abs(native_rgb-adapter_rgb)))}
result['passed']=result['max_pose_error']<1e-5 and result['mean_rgb_error']<1e-5
write_json(args.output/'report.json',result)
print(result)
if not result['passed']:
    raise SystemExit('Native render parity check failed')
