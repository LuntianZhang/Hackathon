"""Reuse image-space masks after SfM refinement, only for identical image bytes."""
import argparse
import hashlib
import shutil
from pathlib import Path
from food3d.paths import resolve_arguments
from food3d.io import initialize_scene, read_json, write_json
from food3d.schemas import load_scene

p=argparse.ArgumentParser()
p.add_argument('--segmentation',required=True,type=Path)
p.add_argument('--scene',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
a=resolve_arguments(p.parse_args())
meta=read_json(a.segmentation/'segmentation.json')
old=Path(meta['source_scene'])
_,old_frames=load_scene(old/'scene.json')
_,new_frames=load_scene(a.scene/'scene.json')
old_by_id={f.frame_id:f for f in old_frames}
for frame in new_frames:
    prior=old_by_id[frame.frame_id]
    if (frame.width,frame.height)!=(prior.width,prior.height):
        raise ValueError('图像尺寸发生变化，需重新分割')
    if hashlib.sha256((a.scene/frame.image_path).read_bytes()).digest()!=hashlib.sha256((old/prior.image_path).read_bytes()).digest():
        raise ValueError('图像字节发生变化，需重新分割')
initialize_scene(a.output)
for kind in ['masks','confidence']:
    meta[kind]={key:str((a.segmentation/path).resolve()) for key,path in meta[kind].items()}
meta['original_source_scene']=str(old)
meta['source_scene']=str(a.scene.resolve())
meta['reuse_validation']='all_registered_image_bytes_and_dimensions_identical'
write_json(a.output/'segmentation.json',meta)
shutil.copy2(a.segmentation/'instances.json',a.output/'instances.json')
print('Reused segmentation after image-byte validation:',len(new_frames))
