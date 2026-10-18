"""Reuse board-background instance tracking after COLMAP undistortion."""

from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from ..data.board_background import undistort_arrays
from ..io import initialize_scene, read_json, write_json
from ..schemas import UNKNOWN_ID, load_scene
from ..visuals import contact_sheet


def reuse_background_segmentation(scene: Path, output: Path, reference_frame: str | None = None) -> dict | None:
    """Return None only when no completed background tracking is available."""
    scene, output = scene.resolve(), output.resolve()
    if not (scene / 'frame_manifest.json').is_file():
        return None
    manifest = read_json(scene / 'frame_manifest.json')
    report = manifest.get('board_background')
    if not report or report.get('status') != 'applied_to_sfm_and_training':
        return None
    import cv2
    import pycolmap
    background = scene / 'board_background'
    evidence = read_json(background / 'vision_evidence.json')
    objects = evidence['objects']
    identifiers = [obj['instance_id'] for obj in objects]
    if not identifiers or len(set(identifiers)) != len(identifiers) or any(
        type(identifier) is not int or not 0 < identifier < UNKNOWN_ID for identifier in identifiers
    ):
        raise ValueError('背景跟踪的水果实例 ID 无效')
    _, frames = load_scene(scene / 'scene.json')
    frame_ids = {frame.frame_id for frame in frames}
    if reference_frame and reference_frame not in frame_ids:
        raise ValueError('参考帧不属于当前场景')
    reference_frame = reference_frame or frames[0].frame_id
    original_names = {entry['frame_id']: Path(entry['image_path']).name for entry in manifest['frames']
                      if entry['accepted']}
    selected = scene / read_json(scene / 'colmap_selection.json')['selected_model']
    if not selected.resolve().is_relative_to(scene) or not selected.is_dir():
        raise ValueError('COLMAP 原始相机模型路径无效')
    native = pycolmap.Reconstruction(selected)
    cameras = {image.name: native.cameras[image.camera_id] for image in native.images.values()}
    if output.exists() and any(output.iterdir()):
        raise ValueError('输出目录已有结果，请选择新的目录')
    # Validate every input before creating a partial result directory.
    for frame in frames:
        name = original_names.get(frame.frame_id)
        if name not in cameras or not (background / 'masks' / f'{frame.frame_id}.png').is_file() or not (
            background / 'confidence' / f'{frame.frame_id}.npy').is_file():
            raise ValueError(f'背景跟踪缺少已注册帧 {frame.frame_id} 的数据')
    initialize_scene(output)
    for folder in ('frames', 'masks', 'confidence', 'debug'):
        (output / folder).mkdir()
    counts = {}
    previews = []
    preview_indices = set(np.linspace(0, len(frames)-1, min(12, len(frames)), dtype=int))
    for index, frame in enumerate(frames):
        fid = frame.frame_id
        with Image.open(background / 'masks' / f'{fid}.png') as image:
            raw_labels = np.asarray(image).astype(np.uint16)
        raw_confidence = np.load(background / 'confidence' / f'{fid}.npy', allow_pickle=False).astype(np.float32)
        camera = cameras[original_names[fid]]
        if raw_labels.shape != (camera.height, camera.width) or raw_confidence.shape != raw_labels.shape:
            raise ValueError(f'背景跟踪与原始相机 {fid} 的像素尺寸不一致')
        if not np.isfinite(raw_confidence).all():
            raise ValueError(f'背景跟踪的置信度包含无效数值：{fid}')
        labels, confidence = undistort_arrays(
            [(raw_labels, cv2.INTER_NEAREST), (raw_confidence, cv2.INTER_LINEAR)],
            camera, frame.K, (frame.width, frame.height))
        unexpected = set(np.unique(labels).tolist()) - {0, UNKNOWN_ID, *identifiers}
        if unexpected:
            raise ValueError(f'背景跟踪包含未知实例 ID：{sorted(unexpected)}')
        confidence = np.clip(confidence, 0, 1)
        confidence[labels == UNKNOWN_ID] = 0
        Image.fromarray(labels).save(output / 'masks' / f'{fid}.png')
        np.save(output / 'confidence' / f'{fid}.npy', confidence.astype(np.float16))
        counts[fid] = {str(identifier): int((labels == identifier).sum()) for identifier in identifiers}
        with Image.open(scene / frame.image_path) as image:
            picture = image.convert('RGB')
        picture.save(output / 'frames' / f'{index:06d}.jpg', quality=95)
        if index in preview_indices:
            colored = np.asarray(picture).copy()
            for identifier in identifiers:
                mask = labels == identifier
                color = np.array([(identifier*71)%255, (identifier*139)%255, (identifier*193)%255])
                colored[mask] = (.6*colored[mask] + .4*color).astype(np.uint8)
            preview = Image.fromarray(colored)
            preview.save(output / 'debug' / f'{fid}.png')
            preview.thumbnail((248, 188))
            previews.append((fid, preview))
        print(f'Background masks remapped {index+1}/{len(frames)}: {fid}', flush=True)
    instances = [{'instance_id': 0, 'name': '背景'}]
    for obj in objects:
        scores = [observation.get('sam_score', 0) for observation in obj['observations']]
        instances.append({'instance_id': obj['instance_id'], 'name': obj['name'],
            'auto_label': obj['label'], 'kind': obj['kind'], 'name_source': 'vision_llm',
            'needs_review': True, 'recognition_views': len(scores),
            'seed_quality': float(np.mean(scores)) if scores else 0})
    write_json(output / 'instances.json', {'schema_version': 1, 'instances': instances,
        'unknown_id': UNKNOWN_ID, 'reference_frame': reference_frame,
        'initialization': 'board_background_reuse', 'assistance': {'mode': 'llm', 'foods': [], 'instances': []},
        'model': evidence['model'], 'keyframes': evidence['keyframes'], 'names_manually_correctable': True})
    shutil.copy2(background / 'vision_evidence.json', output / 'vision_evidence.json')
    contact_sheet(previews, output / 'debug' / 'mask_sheet.png')
    result = {'schema_version': 1, 'source_scene': str(scene), 'frames': len(frames),
        'instance_ids': identifiers, 'masks': {frame.frame_id: f'masks/{frame.frame_id}.png' for frame in frames},
        'confidence': {frame.frame_id: f'confidence/{frame.frame_id}.npy' for frame in frames},
        'pixel_counts': counts, 'initialization': 'board_background_reuse',
        'keyframes': evidence['keyframes'], 'status': 'propagated_requires_identity_review'}
    write_json(output / 'segmentation.json', result)
    return result
