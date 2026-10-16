"""Official Transformers Grounding DINO adapter; downloads weights, not images."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

MODEL_ID = 'IDEA-Research/grounding-dino-tiny'
MODEL_REVISION = 'a2bb814dd30d776dcf7e30523b00659f4f141c71'


def postprocess(processor, outputs, input_ids, image_size, threshold=.30):
    # Installed Transformers 5.15 uses `threshold`; older examples use
    # `box_threshold`. Inspect the actual API instead of guessing a version.
    method = processor.post_process_grounded_object_detection
    keyword = 'threshold' if 'threshold' in inspect.signature(method).parameters else 'box_threshold'
    return method(outputs, input_ids, **{keyword: threshold}, text_threshold=.25,
                  target_sizes=[image_size[::-1]])[0]


def detect(image, foods: list[dict], cache: Path, device='cuda', threshold=.30) -> list[dict]:
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    snapshot = cache/'huggingface'/'models--IDEA-Research--grounding-dino-tiny'/'snapshots'/MODEL_REVISION
    if not (snapshot/'model.safetensors').is_file():
        raise RuntimeError('文字定位权重尚未准备好，请参照 app/ASSISTANCE.md 下载官方模型后重试')
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(snapshot), local_files_only=True, disable_custom_kernels=True).to(device).eval()
    detections = []
    try:
        # Per-category queries make the label association unambiguous even if
        # token-level text decoding returns a partial phrase.
        for food in foods:
            inputs = processor(images=image, text=food['text']+'.', return_tensors='pt').to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            result = postprocess(processor, outputs, inputs.input_ids, image.size, threshold)
            for box, score in zip(result['boxes'].detach().cpu().numpy(), result['scores'].detach().cpu().numpy()):
                box = np.clip(box, [0, 0, 0, 0], [image.width-1, image.height-1]*2)
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                detections.append({'box': box.tolist(), 'grounding_score': float(score),
                                   'name': food['name'], 'text': food['text']})
    finally:
        del model
        torch.cuda.empty_cache()
    return suppress_boxes(detections)


def suppress_boxes(detections: list[dict]) -> list[dict]:
    """Cross-category NMS prevents duplicate IDs for one located object."""
    kept = []
    for candidate in sorted(detections, key=lambda d: d['grounding_score'], reverse=True):
        a = np.asarray(candidate['box'])
        redundant = False
        for existing in kept:
            b = np.asarray(existing['box'])
            intersection = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
            union = (a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-intersection
            if intersection/max(union, 1) > .5:
                redundant = True
                break
        if not redundant:
            kept.append(candidate)
        if len(kept) == 20:
            break
    return sorted(kept, key=lambda d: (d['box'][1], d['box'][0]))
