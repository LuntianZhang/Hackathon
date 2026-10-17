"""Photo recognition worker, optionally bound to verified multi-view geometry."""
import argparse
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, ImageDraw

from food3d.io import read_json, write_json
from food3d.paths import ProjectPaths, resolve_arguments

def reliable_proposals(proposals, classifications, width, height):
    chosen=[]
    for proposal,result in zip(proposals,classifications):
        x,y,w,h=proposal['bbox']
        border=x<=1 or y<=1 or x+w>=width-1 or y+h>=height-1
        if result['similarity'] < .20 or result['margin'] < .01: continue
        # Weak frame-edge fragments often correspond to glare or the table.
        # Keep confident partial food objects; ambiguous omissions are editable.
        if border and result['margin'] < .04: continue
        chosen.append((proposal,result))
    return chosen

def recognize(input_path: Path, output: Path, catalog_path: Path, checkpoint: Path, assistance: dict | None = None, scene: Path | None = None, reference_frame: str | None = None):
    if assistance and assistance.get('mode') == 'llm':
        from food3d.segmentation.hints import validate_assistance
        validate_assistance(assistance)
        from food3d.segmentation.llm_segment import recognize_photo
        return recognize_photo(input_path, output, catalog_path, checkpoint, scene, reference_frame)
    import torch
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from food3d.segmentation.recognition import ClipRecognizer
    from food3d.segmentation.sam2 import select_proposals
    from food3d.segmentation.hints import validate_assistance, candidate_names, display_names, allowed_food
    started = time.monotonic()
    foods = read_json(catalog_path)['foods']
    from food3d.food_catalog import FoodIndex
    food_index=FoodIndex(foods)
    assistance = validate_assistance(assistance, foods)
    if not torch.cuda.is_available():
        raise RuntimeError('识别服务暂不可用，请使用手工添加食物完成记录')
    if output.exists() and any(output.iterdir()):
        raise ValueError('输出目录已有结果，请选择新的目录，避免混入旧掩码')
    output.mkdir(parents=True, exist_ok=True)
    cache = ProjectPaths().cache_root/'models'
    os.environ['HF_HOME'] = str(cache/'huggingface')
    os.environ['TORCH_HOME'] = str(cache/'torch')
    names = display_names(assistance, foods)
    with Image.open(input_path) as source:
        image = ImageOps.exif_transpose(source).convert('RGB')
    image.thumbnail((960, 960))
    image.save(output/'photo.jpg', quality=95)
    manual = assistance['mode'] == 'manual'
    if assistance['mode'] in {'grounded', 'manual'}:
        from food3d.segmentation.prompted import prompted_masks
        proposals = prompted_masks(image, assistance, checkpoint, cache)
    else:
        sam = build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml', str(checkpoint), device='cuda', apply_postprocessing=False)
        generator = SAM2AutomaticMaskGenerator(sam, points_per_side=24, points_per_batch=16,
                        pred_iou_thresh=.85, stability_score_thresh=.92, min_mask_region_area=0)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            proposals = generator.generate(np.asarray(image))
        del generator, sam
        torch.cuda.empty_cache()
    proposals = [proposal for proposal in proposals if .001 <= proposal['area']/(image.width*image.height) <= .75]
    classifications = []
    if manual:
        # An explicit user name is preserved; do not silently replace it with CLIP.
        chosen = [(p, {'label': p['seed']['name'], 'similarity': None, 'margin': None}) for p in proposals]
    else:
        recognizer = ClipRecognizer(candidate_names(assistance, food_index.labels()), cache)
        for start in range(0, len(proposals), 16):
            classifications.extend(recognizer.classify(image, [p['segmentation'] for p in proposals[start:start+16]]))
        # Reject near-ties before NMS; scores are not probabilities.
        reliable = reliable_proposals(proposals,classifications,image.width,image.height)
        reliable = [(p,c) for p,c in reliable if allowed_food(assistance, c['label'])]
        chosen = select_proposals([p for p,_ in reliable], [c for _,c in reliable],
                     image.width*image.height, max_instances=20, max_fraction=.75)
        del recognizer
        torch.cuda.empty_cache()
    entries = []
    overlay = np.asarray(image).copy()
    for index, (proposal, classification) in enumerate(chosen, 1):
        identifier = proposal['seed']['instance_id'] if manual else index
        mask_path = output/f'mask_{identifier}.png'
        Image.fromarray(proposal['segmentation'].astype(np.uint8)*255).save(mask_path)
        label = classification['label']
        name = label if manual else names.get(label, label)
        key = food_index.match(name) or food_index.match(label)
        entries.append({'instance_id': identifier, 'name': name,
                        'catalog_key': key, 'mask_path': str(mask_path.resolve()),
                        'nutrient_status':food_index.foods[key].get('nutrient_status','verified') if key else 'unavailable',
                        'recognition_score': classification['similarity'], 'recognition_margin': classification['margin'],
                        'bbox': proposal['bbox'], 'needs_review': not manual,
                        'name_source': 'manual_prompt' if manual else 'automatic',
                        'recognition_label': None if manual else label,
                        'sam_score': proposal['predicted_iou'],
                        'grounding_score': proposal.get('seed', {}).get('grounding_score')})
        color = np.array([(identifier*71)%220+30, (identifier*139)%220+30, (identifier*193)%220+30])
        mask = proposal['segmentation']
        overlay[mask] = (.55*overlay[mask]+.45*color).astype(np.uint8)
    preview = Image.fromarray(overlay)
    draw = ImageDraw.Draw(preview)
    for entry in entries:
        x,y,w,h = entry['bbox']
        draw.text((x,y), f"ID {entry['instance_id']}", fill='white', stroke_width=2, stroke_fill='black')
    preview.save(output/'overlay.png')
    diagnostics = []
    for proposal, classification in zip(proposals, classifications):
        retained = any(proposal is p for p,_ in chosen)
        diagnostics.append({'bbox': proposal['bbox'], 'selected': retained,
            'label': classification['label'], 'similarity': classification['similarity'],
            'margin': classification['margin'], 'scores': classification['scores'],
            'text_detection': proposal.get('seed'),
            'reason': ('accepted' if retained else 'outside_hint_list' if not allowed_food(assistance, classification['label'])
                       else 'low_score_margin_edge_or_duplicate')})
    result = {'photo': str((output/'photo.jpg').resolve()), 'width': image.width, 'height': image.height,
              'items': entries, 'three_d_available': False,
              'assistance': assistance, 'elapsed_seconds': round(time.monotonic()-started, 2),
              'overlay': str((output/'overlay.png').resolve()),
              'proposals': len(proposals), 'rejected': len(proposals)-len(entries),
              'missing_hints': [f['name'] for f in assistance['foods'] if f['text'] not in
                                {e['recognition_label'] for e in entries}],
              'candidate_diagnostics': diagnostics,
              'recognition': 'SAM2.1 pretrained + CLIP ViT-B-32; scores are similarities, not probabilities'}
    write_json(output/'recognition.json', result)
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--catalog', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('.tools/weights/sam2.1_hiera_small.pt'))
    parser.add_argument('--assistance', type=Path, help='辅助方式、食物候选和归一化点击/框选 JSON')
    parser.add_argument('--scene', type=Path, help='大模型联合识别的已验证场景')
    parser.add_argument('--reference-frame')
    args = resolve_arguments(parser.parse_args())
    recognize(args.input, args.output, args.catalog, args.checkpoint,
              read_json(args.assistance) if args.assistance else None, args.scene, args.reference_frame)
