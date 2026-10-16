from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .io import initialize_scene, read_json, write_json
from .schemas import UNKNOWN_ID, load_scene
from .viewer.app import describe_click
from .viewer.renderer import GaussianRenderer, resize_camera
from .visuals import contact_sheet


def evaluate(bundle: Path, output: Path, width: int = 640) -> dict:
    from scipy.ndimage import distance_transform_edt
    bundle, output = bundle.resolve(), output.resolve()
    if width < 32:
        raise ValueError("评估宽度至少为 32")
    renderer = GaussianRenderer(bundle)
    fusion = read_json(bundle / "semantics" / "fusion.json")
    segmentation = Path(fusion["segmentation"])
    segment_data = read_json(segmentation / "segmentation.json")
    names = {entry["instance_id"]: entry["name"] for entry in read_json(bundle / "semantics" / "instances.json")["instances"]}
    scene_data, cameras = load_scene(bundle / "scene.json")
    by_id = {camera.frame_id: camera for camera in cameras}
    holdout = fusion["holdout_views"]
    if set(holdout) & set(fusion["voting_views"]):
        raise ValueError("保留视角参与了投票，不能作为独立语义一致性检查")
    initialize_scene(output)
    per_instance = {instance_id: {"intersection": 0, "union": 0} for instance_id in names if instance_id > 0}
    clicks, per_view, previews = [], [], []
    for frame_id in holdout:
        frame = resize_camera(by_id[frame_id], width)
        result = renderer.render(frame)
        with Image.open(bundle / frame.image_path) as image:
            source = image.convert("RGB").resize((frame.width, frame.height), Image.Resampling.LANCZOS)
        with Image.open(segmentation / segment_data["masks"][frame_id]) as image:
            expected = np.asarray(image.resize((frame.width, frame.height), Image.Resampling.NEAREST)).astype(np.uint16)
        prediction = result["instance_ids"]
        valid = expected != UNKNOWN_ID
        for instance_id, counts in per_instance.items():
            target, estimate = (expected == instance_id) & valid, (prediction == instance_id) & valid
            counts["intersection"] += int((target & estimate).sum())
            counts["union"] += int((target | estimate).sum())
            if target.any():
                distance = distance_transform_edt(target)
                y, x = np.unravel_index(np.argmax(distance), distance.shape)
                actual, description = describe_click(prediction, result["confidence"], int(x), int(y), names)
                clicks.append({"frame_id": frame_id, "expected_id": instance_id, "actual_id": actual, "x": int(x), "y": int(y), "confidence": float(result["confidence"][y, x]), "correct": actual == instance_id, "description": description})
        background = expected == 0
        if background.any():
            distance = distance_transform_edt(background)
            y, x = np.unravel_index(np.argmax(distance), distance.shape)
            actual, description = describe_click(prediction, result["confidence"], int(x), int(y), names)
            clicks.append({"frame_id": frame_id, "expected_id": 0, "actual_id": actual, "x": int(x), "y": int(y), "correct": actual == 0 and description == "背景（Instance 0）", "description": description})
        error = (np.asarray(source, dtype=float)/255 - result["rgb"].astype(float)/255)**2
        mse = float(error.mean())
        foreground = (expected > 0) & valid
        fg_mse = float(error[foreground].mean()) if foreground.any() else None
        per_view.append({"frame_id": frame_id, "psnr_db": float(-10*np.log10(max(mse, 1e-12))), "foreground_psnr_db": float(-10*np.log10(max(fg_mse, 1e-12))) if fg_mse is not None else None, "food_identity_pixel_accuracy": float((prediction[foreground] == expected[foreground]).mean()) if foreground.any() else None, "background_food_pixel_fraction": float(((prediction[background] > 0) & (prediction[background] != UNKNOWN_ID)).mean()) if background.any() else None})
        Image.fromarray(result["rgb"]).save(output / f"{frame_id}_render.png")
        np.save(output / f"{frame_id}_instance_ids.npy", prediction)
        rendered = Image.fromarray(result["rgb"])
        source.thumbnail((248, 188))
        rendered.thumbnail((248, 188))
        previews.extend([(frame_id + " source", source), (frame_id + " 3D render", rendered)])
        print(f"Evaluate held-out view: {frame_id}", flush=True)
    iou = {str(instance_id): counts["intersection"]/counts["union"] if counts["union"] else None for instance_id, counts in per_instance.items()}
    food_clicks = [click for click in clicks if click["expected_id"] > 0]
    background_clicks = [click for click in clicks if click["expected_id"] == 0]
    foreground_false_positives = sum(click['actual_id'] > 0 for click in background_clicks)
    report = {"schema_version": 1, "reference_type": "SAM2_heldout_view_consistency_not_human_ground_truth", "model_sha256": renderer.metadata["model_sha256"], "heldout_views": holdout, "voting_views": fusion["voting_views"], "instance_iou": iou, "mean_instance_iou": float(np.mean([value for value in iou.values() if value is not None])), "food_click_accuracy": sum(click["correct"] for click in food_clicks)/max(len(food_clicks), 1), "food_click_count": len(food_clicks), "background_click_accuracy": sum(click["correct"] for click in background_clicks)/max(len(background_clicks), 1), "background_click_count": len(background_clicks), "mean_psnr_db": float(np.mean([view["psnr_db"] for view in per_view])), "per_view": per_view, "clicks": clicks, "limitation": "输入为渲染图片；指标衡量对 SAM 2 的跨视角一致性，尚未用实物照片和人工逐像素真值验证。"}
    contact_sheet(previews, output / "comparison_sheet.png")
    report['background_no_food_name_accuracy'] = 1-foreground_false_positives/max(len(background_clicks),1)
    report['background_food_false_positive_count'] = foreground_false_positives
    report['unknown_gaussian_fraction'] = fusion['unknown_fraction']
    report['mean_foreground_psnr_db'] = float(np.mean([view['foreground_psnr_db'] for view in per_view if view['foreground_psnr_db'] is not None]))
    report['input_source'] = scene_data.get('metadata', {}).get('input_source', 'unspecified')
    report['limitation'] = ('指标衡量对 SAM 2 的跨视角一致性；尚未用人工逐像素真值验证。'
                            '保留视角不参与语义投票，但用于 SfM、SAM 传播和开发检查。')
    write_json(output / "report.json", report)
    return {key: report[key] for key in ["reference_type", "mean_instance_iou", "food_click_accuracy", "background_click_accuracy", "mean_psnr_db"]}
