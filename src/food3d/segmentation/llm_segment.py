"""Joint vision recognition, multi-frame SAM anchors, and bidirectional tracking."""
from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from ..io import initialize_scene, read_json, write_json
from ..paths import ProjectPaths
from ..schemas import UNKNOWN_ID, load_scene
from ..visuals import contact_sheet
from .vision_llm import VisionSettings, identify_views, select_keyframes


def anchor_masks(views, evidence, checkpoint):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml', str(checkpoint),
                       device='cuda', apply_postprocessing=False)
    predictor = SAM2ImagePredictor(model)
    anchors = {}
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for fid, image in views:
                predictor.set_image(np.asarray(image.convert('RGB')))
                for obj in evidence['objects']:
                    observation = next((x for x in obj['observations'] if x['frame_id'] == fid), None)
                    if observation is None:
                        continue
                    box = np.asarray(observation['box'])*([image.width-1, image.height-1]*2)
                    masks, scores, _ = predictor.predict(box=box.astype(np.float32), multimask_output=True)
                    best = int(np.argmax(scores))
                    mask = masks[best].astype(bool)
                    # Reject degenerate masks and proposals that flood the image.
                    if mask.sum() < 16 or mask.mean() > .8 or not np.isfinite(scores[best]):
                        continue
                    anchors.setdefault(fid, {})[obj['instance_id']] = mask
                    observation['sam_score'] = float(scores[best])
    finally:
        del predictor, model
        torch.cuda.empty_cache()
    supported = {identifier for frame in anchors.values() for identifier in frame}
    evidence['objects'] = [obj for obj in evidence['objects'] if obj['instance_id'] in supported]
    if not supported:
        raise ValueError('大模型定位未得到有效分割，请更换视角或其他识别方式')
    return anchors


def label_scores(scores, identifiers):
    top = np.argmax(scores, axis=0)
    best = np.take_along_axis(scores, top[None], axis=0)[0]
    labels = np.asarray(identifiers, np.uint16)[top]
    labels[best <= 0] = 0
    probability = 1/(1+np.exp(-np.clip(scores, -30, 30)))
    first = probability.max(axis=0)
    second = np.partition(probability, -2, axis=0)[-2] if len(identifiers) > 1 else np.zeros_like(first)
    ambiguous = (labels > 0) & (first-second < .1)
    labels[ambiguous] = UNKNOWN_ID
    confidence = np.where(labels == 0, 1-first, first)
    confidence[ambiguous] = 0
    return labels, confidence.astype(np.float16)


def track_anchors(frames, anchors, output, checkpoint, identifiers, reverse_pass=True):
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml',
                                         str(checkpoint), device='cuda', apply_postprocessing=False)
    by_id = {identifier: [i for i, f in enumerate(frames) if identifier in anchors.get(f.frame_id, {})]
             for identifier in identifiers}
    temporary = output/'tracking_temporary'; temporary.mkdir()
    counts = {}
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            state = predictor.init_state(video_path=str(output/'frames'),
                                         offload_video_to_cpu=True, offload_state_to_cpu=True)
            # All conditioning frames are registered before propagation begins.
            for i, frame in enumerate(frames):
                for identifier, mask in anchors.get(frame.frame_id, {}).items():
                    predictor.add_new_mask(state, frame_idx=i, obj_id=identifier, mask=mask)
            for reverse in (False, True) if reverse_pass else (False,):
                start = len(frames)-1 if reverse else 0
                for index, obj_ids, logits in predictor.propagate_in_video(state, start_frame_idx=start, reverse=reverse):
                    raw = logits[:, 0].float().cpu().numpy()
                    order = {int(identifier): i for i, identifier in enumerate(obj_ids)}
                    scores = np.stack([raw[order[identifier]] if identifier in order else
                                       np.full(raw.shape[1:], -30, np.float32) for identifier in identifiers])
                    path = temporary/f'{index:06d}.npy'
                    if not reverse and reverse_pass:
                        np.save(path, scores.astype(np.float16))
                        continue
                    if reverse_pass:
                        if not path.is_file():
                            raise RuntimeError('多视图传播缺少正向结果')
                        forward = np.load(path, allow_pickle=False).astype(np.float32)
                    for channel, identifier in enumerate(identifiers):
                        if reverse_pass:
                            past = min((index-i for i in by_id[identifier] if i <= index), default=float('inf'))
                            future = min((i-index for i in by_id[identifier] if i >= index), default=float('inf'))
                            if past <= future:
                                scores[channel] = forward[channel]
                        anchor = anchors.get(frames[index].frame_id, {}).get(identifier)
                        if anchor is not None:
                            scores[channel] = np.where(anchor, 10, -10)
                    labels, confidence = label_scores(scores, identifiers)
                    fid = frames[index].frame_id
                    Image.fromarray(labels).save(output/'masks'/f'{fid}.png')
                    np.save(output/'confidence'/f'{fid}.npy', confidence)
                    counts[fid] = {str(identifier): int((labels == identifier).sum()) for identifier in identifiers}
                    if reverse_pass:
                        path.unlink()
                    print(f'LLM multi-anchor SAM 2 frame {index+1}/{len(frames)}', flush=True)
            if len(counts) != len(frames):
                raise RuntimeError('多视图传播未覆盖全部相机')
            del state
    finally:
        shutil.rmtree(temporary)
        del predictor
        torch.cuda.empty_cache()
    return counts


def segment_llm(scene: Path, output: Path, checkpoint: Path, reference_frame=None):
    from .reuse_background import reuse_background_segmentation
    reused = reuse_background_segmentation(scene, output, reference_frame)
    if reused is not None:
        return reused
    import torch
    settings = VisionSettings.load()
    if not torch.cuda.is_available():
        raise RuntimeError('大模型精细分割需要本机 CUDA 服务')
    scene, output = scene.resolve(), output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('输出目录已有结果，请选择新的目录')
    _, frames = load_scene(scene/'scene.json')
    if len({(f.width, f.height) for f in frames}) != 1:
        raise ValueError('多视图传播需要一致的图像尺寸')
    if reference_frame and reference_frame not in {f.frame_id for f in frames}:
        raise ValueError('参考帧不属于当前场景')
    reference_frame = reference_frame or frames[0].frame_id
    indices = select_keyframes(frames, reference_frame, settings.max_views)
    views = []
    for index in indices:
        frame = frames[index]
        with Image.open(scene/frame.image_path) as source:
            views.append((frame.frame_id, source.convert('RGB')))
    evidence = identify_views(views, settings, request_root=output.parent/'agent_requests')
    anchors = anchor_masks(views, evidence, checkpoint)
    initialize_scene(output)
    for folder in ('frames', 'masks', 'confidence', 'debug', 'anchors'):
        (output/folder).mkdir()
    for index, frame in enumerate(frames):
        with Image.open(scene/frame.image_path) as source:
            source.convert('RGB').save(output/'frames'/f'{index:06d}.jpg', quality=95)
    for fid, objects in anchors.items():
        for identifier, mask in objects.items():
            Image.fromarray(mask.astype(np.uint8)*255).save(output/'anchors'/f'{fid}_{identifier}.png')
    instances = [{'instance_id': 0, 'name': '背景'}]+[
        {'instance_id': obj['instance_id'], 'name': obj['name'], 'auto_label': obj['label'],
         'kind': obj['kind'], 'name_source': 'vision_llm', 'needs_review': True,
         'recognition_views': sum(obj['instance_id'] in x for x in anchors.values()),
         'seed_quality': float(np.mean([x.get('sam_score', 0) for x in obj['observations']]))}
        for obj in evidence['objects']]
    identifiers = [obj['instance_id'] for obj in evidence['objects']]
    write_json(output/'vision_evidence.json', evidence)
    write_json(output/'instances.json', {'schema_version': 1, 'instances': instances,
        'unknown_id': UNKNOWN_ID, 'reference_frame': reference_frame, 'initialization': 'llm_multiview',
        'assistance': {'mode': 'llm', 'foods': [], 'instances': []}, 'model': evidence['model'],
        'keyframes': evidence['keyframes'], 'names_manually_correctable': True})
    counts = track_anchors(frames, anchors, output, checkpoint, identifiers)
    previews = []
    for index in sorted(set(indices+np.linspace(0, len(frames)-1, min(8, len(frames)), dtype=int).tolist())):
        frame = frames[index]
        with Image.open(scene/frame.image_path) as source:
            pixels = np.asarray(source.convert('RGB')).copy()
        with Image.open(output/'masks'/f'{frame.frame_id}.png') as source:
            labels = np.asarray(source)
        for identifier in identifiers:
            mask = labels == identifier
            color = np.array([(identifier*71)%255, (identifier*139)%255, (identifier*193)%255])
            pixels[mask] = (.6*pixels[mask]+.4*color).astype(np.uint8)
        preview = Image.fromarray(pixels); preview.save(output/'debug'/f'{frame.frame_id}.png')
        preview.thumbnail((248, 188)); previews.append((frame.frame_id, preview))
    contact_sheet(previews, output/'debug/mask_sheet.png')
    result = {'schema_version': 1, 'source_scene': str(scene), 'frames': len(frames),
        'instance_ids': identifiers, 'masks': {f.frame_id: f'masks/{f.frame_id}.png' for f in frames},
        'confidence': {f.frame_id: f'confidence/{f.frame_id}.npy' for f in frames},
        'pixel_counts': counts, 'initialization': 'llm_multiview', 'keyframes': evidence['keyframes'],
        'anchor_frames_per_instance': {str(i): [fid for fid, objects in anchors.items() if i in objects]
                                       for i in identifiers},
        'status': 'propagated_requires_identity_review'}
    write_json(output/'segmentation.json', result)
    return result


def recognize_photo(input_path, output, catalog_path, checkpoint, scene=None, reference_frame=None):
    """Photo-compatible output; bound photos use the complete scene once."""
    import time
    import torch
    from PIL import ImageDraw, ImageOps
    started = time.monotonic()
    settings = VisionSettings.load()
    if not torch.cuda.is_available():
        raise RuntimeError('大模型精细分割需要本机 CUDA 服务')
    if output.exists() and any(output.iterdir()):
        raise ValueError('输出目录已有结果，请选择新的目录')
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(input_path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    image.thumbnail((960, 960)); image.save(output/'photo.jpg', quality=95)
    segmentation_path = None
    if scene is not None:
        segmentation_path = output/'multiview'
        segment_llm(Path(scene), segmentation_path, checkpoint, reference_frame)
        reference_frame = read_json(segmentation_path/'instances.json')['reference_frame']
        evidence = read_json(segmentation_path/'vision_evidence.json')
        with Image.open(segmentation_path/'masks'/f'{reference_frame}.png') as source:
            labels = np.asarray(source.resize(image.size, Image.Resampling.NEAREST))
        masks = {obj['instance_id']: labels == obj['instance_id'] for obj in evidence['objects']}
    else:
        evidence = identify_views([('photo', image)], settings, request_root=output/'agent_requests')
        masks = anchor_masks([('photo', image)], evidence, checkpoint).get('photo', {})
    catalog = read_json(catalog_path)['foods']
    from ..food_catalog import FoodIndex
    food_index=FoodIndex(catalog)
    entries, pixels = [], np.asarray(image).copy()
    for obj in evidence['objects']:
        identifier = obj['instance_id']; mask = masks.get(identifier, np.zeros((image.height, image.width), bool))
        path = output/f'mask_{identifier}.png'; Image.fromarray(mask.astype(np.uint8)*255).save(path)
        y, x = np.nonzero(mask)
        box = [int(x.min()), int(y.min()), int(np.ptp(x)+1), int(np.ptp(y)+1)] if len(x) else [0, 0, 0, 0]
        # Packages and nonfood test objects must not inherit edible volume/nutrition.
        key = food_index.match(obj['name']) or food_index.match(obj['label'])
        if obj['kind'] != 'food': key = None
        entries.append({'instance_id': identifier, 'name': obj['name'], 'catalog_key': key,
            'nutrient_status':food_index.foods[key].get('nutrient_status','verified') if key else 'unavailable',
            'mask_path': str(path.resolve()), 'bbox': box, 'needs_review': True, 'name_source': 'vision_llm',
            'recognition_label': obj['label'], 'kind': obj['kind'], 'visible_in_reference': bool(len(x)),
            'recognition_score': float(np.mean([o['confidence'] for o in obj['observations']])),
            'recognition_margin': None, 'sam_score': float(np.mean([o.get('sam_score', 0) for o in obj['observations']])),
            'recognition_views': len(obj['observations'])})
        color = np.array([(identifier*71)%220+30, (identifier*139)%220+30, (identifier*193)%220+30])
        pixels[mask] = (.55*pixels[mask]+.45*color).astype(np.uint8)
    preview = Image.fromarray(pixels); draw = ImageDraw.Draw(preview)
    for entry in entries:
        if entry['visible_in_reference']:
            draw.text(tuple(entry['bbox'][:2]), f"ID {entry['instance_id']}", fill='white', stroke_width=2, stroke_fill='black')
    preview.save(output/'overlay.png'); write_json(output/'vision_evidence.json', evidence)
    result = {'photo': str((output/'photo.jpg').resolve()), 'width': image.width, 'height': image.height,
        'items': entries, 'three_d_available': False, 'assistance': {'mode': 'llm', 'foods': [], 'instances': []},
        'elapsed_seconds': round(time.monotonic()-started, 2), 'overlay': str((output/'overlay.png').resolve()),
        'proposals': len(entries), 'rejected': 0, 'missing_hints': [], 'candidate_diagnostics': [],
        'recognition': 'Vision LLM joint identities + SAM 2 multi-frame anchors; scores are not calibrated probabilities',
        'model': evidence['model'], 'keyframes': evidence['keyframes'],
        'segmentation_path': str(segmentation_path.resolve()) if segmentation_path else None}
    write_json(output/'recognition.json', result)
    return result
