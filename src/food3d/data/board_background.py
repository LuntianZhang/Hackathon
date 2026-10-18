"""Exclude background before SfM, preserving board and full object silhouettes.

Input masks use the original pixel grid, without poses, guessed intrinsics,
resizing or perspective rectification.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image

from ..io import read_json, write_json
from ..visuals import contact_sheet
from ..segmentation.board_scope import board_polygon


def foreground_mask(size, polygon, labels):
    """Union the printed board and complete tracked objects, including overhangs."""
    import cv2
    width,height=size
    if labels.shape!=(height,width): raise ValueError('定位板和物体掩码尺寸不一致')
    mask=np.zeros((height,width),np.uint8)
    cv2.fillConvexPoly(mask,np.round(polygon).astype(np.int32),255)
    mask[labels>0]=255  # Includes uncertain overlaps between foreground objects.
    return cv2.dilate(mask,np.ones((3,3),np.uint8))


def prepare_background(scene: Path, checkpoint: Path, minimum=2):
    """Detect boards, jointly recognize unposed views, then track foreground."""
    from ..segmentation.llm_segment import anchor_masks,track_anchors
    from ..segmentation.vision_llm import VisionSettings,identify_views
    if minimum<2:raise ValueError('有效定位板视角数至少为 2')
    scene=scene.resolve();manifest=read_json(scene/'frame_manifest.json')
    output=scene/'board_background'
    if output.exists(): raise ValueError('背景处理目录已存在，请使用新的分析目录')
    accepted=[f for f in manifest['frames'] if f['accepted']]
    frames=[];regions={};excluded=[]
    for frame in accepted:
        with Image.open(scene/frame['image_path']) as image:
            region=board_polygon(image);width,height=image.size
        if region is None:
            excluded.append(frame);continue
        regions[frame['frame_id']]=region
        frames.append(SimpleNamespace(frame_id=frame['frame_id'],image_path=frame['image_path'],width=width,height=height))
    if len(frames)<minimum:
        raise ValueError(f'仅 {len(frames)} 张照片检测到完整定位板，至少需要 {minimum} 张；请露出至少 3 个标记，或关闭排除背景')
    if len({(f.width,f.height) for f in frames})!=1:
        raise ValueError('背景跟踪需要相同尺寸的照片')
    output.mkdir()
    for folder in ('frames','masks','confidence','feature_masks','debug'):(output/folder).mkdir()
    settings=VisionSettings.load()
    indices=np.linspace(0,len(frames)-1,min(settings.max_views,len(frames)),dtype=int).tolist()
    views=[]
    for index in indices:
        frame=frames[index]
        with Image.open(scene/frame.image_path) as image:views.append((frame.frame_id,image.convert('RGB')))
    evidence=identify_views(views,settings,request_root=scene/'agent_requests_background')
    anchors=anchor_masks(views,evidence,checkpoint)
    for index,frame in enumerate(frames):
        shutil.copyfile(scene/frame.image_path,output/'frames'/f'{index:06d}.jpg')
    identifiers=[obj['instance_id'] for obj in evidence['objects']]
    # Anchors at both ends plus intermediate observations let the forward pass
    # cover this turntable sequence without a second full-video GPU traversal.
    forward_only = all(identifier in anchors.get(frames[0].frame_id,{}) and
        identifier in anchors.get(frames[-1].frame_id,{}) and
        sum(identifier in objects for objects in anchors.values()) >= 3
        for identifier in identifiers)
    if forward_only:
        track_anchors(frames,anchors,output,checkpoint,identifiers,reverse_pass=False)
    else:
        track_anchors(frames,anchors,output,checkpoint,identifiers)
    previews=[];records={}
    for index,frame in enumerate(frames):
        with Image.open(output/'masks'/f'{frame.frame_id}.png') as image:labels=np.asarray(image)
        region=regions[frame.frame_id]
        mask=foreground_mask((frame.width,frame.height),region['polygon_pixels'],labels)
        path=output/'feature_masks'/(Path(frame.image_path).name+'.png');Image.fromarray(mask).save(path)
        records[frame.frame_id]={'mask_path':path.relative_to(scene).as_posix(),
            'board':region,'retained_fraction':float((mask>0).mean())}
        if index in indices:
            with Image.open(scene/frame.image_path) as image:pixels=np.asarray(image.convert('RGB')).copy()
            pixels[mask==0]=255;preview=Image.fromarray(pixels);preview.thumbnail((300,240))
            previews.append((frame.frame_id,preview))
    contact_sheet(previews,output/'debug/preview.jpg')
    write_json(output/'vision_evidence.json',evidence)
    result={'enabled':True,'method':'board_markers_and_tracked_full_objects',
        'tracking_passes':'forward_only' if forward_only else 'bidirectional',
        'status':'input_masks_ready','processed_frames':len(frames),'excluded_frames':len(excluded),
        'excluded_frame_ids':[f['frame_id'] for f in excluded],
        'feature_mask_path':'board_background/feature_masks','frames':records,
        'pixel_policy':'original_grid_no_warp_union_of_board_and_full_objects',
        'uses_camera_poses':False,'preview':'board_background/debug/preview.jpg'}
    # Leave the source manifest intact until all masks exist. Unverified frames
    # must not enter COLMAP after the user requested background exclusion.
    for frame in excluded:
        destination=output/'excluded_frames'/Path(frame['image_path']).name
        destination.parent.mkdir(exist_ok=True)
        shutil.move(scene/frame['image_path'],destination)
        frame.update(accepted=False,rejection_reason='positioning_board_not_detected',
                     image_path=destination.relative_to(scene).as_posix())
    manifest['board_background']=result
    write_json(scene/'frame_manifest.json',manifest);write_json(output/'report.json',result)
    return result


def feature_mask_path(scene, manifest):
    """Validate the complete mask set before either COLMAP backend reads it."""
    report=manifest.get('board_background')
    if not report or not report.get('enabled'):return None
    root=(scene/report['feature_mask_path']).resolve()
    if not root.is_relative_to(scene.resolve()):raise ValueError('背景掩码目录必须属于当前场景')
    for frame in manifest['frames']:
        if not frame['accepted']:continue
        path=root/(Path(frame['image_path']).name+'.png')
        if not path.is_file():raise ValueError('背景掩码缺失，请重新进行背景跟踪')
        with Image.open(scene/frame['image_path']) as image,Image.open(path) as mask:
            if image.size!=mask.size:raise ValueError('背景掩码与输入照片尺寸不一致')
    return root


def undistort_arrays(arrays, original_camera, target_k, size):
    """Remap related raw images with one COLMAP camera projection per strip."""
    import cv2
    width,height=size;k=np.asarray(target_k)
    if not arrays or any(array.ndim != 2 or array.shape != arrays[0][0].shape for array,_ in arrays):
        raise ValueError('去畸变输入必须是同尺寸的二维图像')
    results=[np.zeros((height,width),array.dtype) for array,_ in arrays]
    for start in range(0,height,128):
        y,x=np.mgrid[start:min(start+128,height),:width]
        rays=np.column_stack(((x.ravel()-k[0,2])/k[0,0],
            (y.ravel()-k[1,2])/k[1,1],np.ones(x.size)))
        # COLMAP pixel centres have a half-pixel offset from our integer grid.
        pixels=original_camera.img_from_cam(rays)-.5;shape=x.shape
        map_x=pixels[:,0].reshape(shape).astype(np.float32)
        map_y=pixels[:,1].reshape(shape).astype(np.float32)
        for result,(array,interpolation) in zip(results,arrays):
            result[start:start+shape[0]]=cv2.remap(array,map_x,map_y,interpolation,
                borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return results


def undistort_mask(mask, original_camera, target_k, size):
    """Map a raw mask through the actual COLMAP distortion model and K."""
    import cv2
    return undistort_arrays([(mask,cv2.INTER_NEAREST)],original_camera,target_k,size)[0]


def apply_background(scene, manifest, frames, original_model):
    """Create white-background training images while preserving image geometry."""
    report=manifest.get('board_background')
    if not report or not report.get('enabled'):return frames
    import pycolmap
    if original_model is None:raise ValueError('背景掩码去畸变缺少原始相机模型')
    native=pycolmap.Reconstruction(original_model)
    cameras={image.name:native.cameras[image.camera_id] for image in native.images.values()}
    mask_root=feature_mask_path(scene,manifest)
    output=scene/'board_background';(output/'images').mkdir();(output/'undistorted_masks').mkdir()
    updated=[]
    for frame in frames:
        name=Path(frame.image_path).name
        with Image.open(mask_root/(name+'.png')) as image:raw_mask=np.asarray(image.convert('L'))
        mask=undistort_mask(raw_mask,cameras[name],frame.K,(frame.width,frame.height))
        Image.fromarray(mask).save(output/'undistorted_masks'/f'{frame.frame_id}.png')
        with Image.open(scene/frame.image_path) as image:pixels=np.asarray(image.convert('RGB')).copy()
        pixels[mask==0]=255
        path=output/'images'/f'{frame.frame_id}.png';Image.fromarray(pixels).save(path)
        updated.append(replace(frame,image_path=path.relative_to(scene).as_posix()))
    report.update(status='applied_to_sfm_and_training',registered_frames=len(updated))
    write_json(output/'report.json',report);write_json(scene/'frame_manifest.json',manifest)
    return updated
