"""Shared text/manual initialization producing real SAM 2 masks."""
from __future__ import annotations

import numpy as np


def prompted_masks(image, assistance, checkpoint, cache):
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if assistance['mode'] == 'grounded':
        from .grounding import detect
        seeds = detect(image, assistance['foods'], cache)
        seeds = [{'instance_id': index, **seed} for index, seed in enumerate(seeds, 1)]
    else:
        seeds = []
        for source in assistance['instances']:
            seed = dict(source)
            if 'box' in seed:
                seed['box'] = (np.asarray(seed['box'])*([image.width-1, image.height-1]*2)).tolist()
            else:
                seed['point'] = (np.asarray(seed['point'])*[image.width-1, image.height-1]).tolist()
            seeds.append(seed)
    if not seeds:
        return []
    model = build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml', str(checkpoint),
                      device='cuda', apply_postprocessing=False)
    predictor = SAM2ImagePredictor(model)
    proposals = []
    try:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            predictor.set_image(np.array(image))
            for seed in seeds:
                kwargs = ({'box': np.asarray(seed['box'], dtype=np.float32)} if 'box' in seed else
                          {'point_coords': np.asarray([seed['point']], dtype=np.float32),
                           'point_labels': np.array([1], dtype=np.int32)})
                masks, scores, _ = predictor.predict(**kwargs, multimask_output=True)
                best = int(np.argmax(scores))
                mask = masks[best].astype(bool)
                y, x = np.nonzero(mask)
                if len(x) < 16:
                    continue
                proposals.append({'segmentation': mask, 'area': len(x),
                    'bbox': [int(x.min()), int(y.min()), int(x.max()-x.min()+1), int(y.max()-y.min()+1)],
                    'predicted_iou': float(scores[best]), 'stability_score': 1.0, 'seed': seed})
    finally:
        del predictor, model
        torch.cuda.empty_cache()
    return proposals
