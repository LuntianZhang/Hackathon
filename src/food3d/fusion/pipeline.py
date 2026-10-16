from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from ..io import initialize_scene, read_json, write_json
from ..schemas import UNKNOWN_ID, load_scene
from ..viewer.renderer import GaussianRenderer, resize_camera
from .visibility import splat_contributions, pixel_surface_depths
from .voting import finalize_votes, weight_view_evidence


def fuse(bundle: Path, segmentation: Path, max_views: int = 40, render_width: int = 480, minimum_views: int = 3, depth_reference: str = 'surface') -> dict:
    import torch
    from scipy.ndimage import distance_transform_edt
    bundle, segmentation = bundle.resolve(), segmentation.resolve()
    if max_views < 3 or render_width < 32 or minimum_views < 1:
        raise ValueError("融合参数无效")
    if depth_reference not in {'surface','expected'}:
        raise ValueError('Depth reference must be surface or expected')
    renderer = GaussianRenderer(bundle)
    _, frames = load_scene(bundle / "scene.json")
    segment_data = read_json(segmentation / "segmentation.json")
    if Path(segment_data["source_scene"]).resolve() != Path(renderer.metadata["source_scene"]).resolve():
        raise ValueError("分割与三维模型来自不同场景")
    instances = read_json(segmentation / "instances.json")
    instance_values = np.array(sorted(entry["instance_id"] for entry in instances["instances"]), dtype=np.uint16)
    if len(instance_values) < 2 or instance_values[0] != 0:
        raise ValueError("需要背景和至少一个食物实例")
    n, classes = len(renderer.model.means), len(instance_values)
    votes = np.zeros((n, classes), dtype=np.float32)
    support = np.zeros((n, classes), dtype=np.uint16)
    holdout = list(range(0, len(frames), 10))
    available = [index for index in range(len(frames)) if index not in holdout]
    indices = [available[index] for index in np.linspace(0, len(available)-1, min(max_views, len(available)), dtype=int)]
    per_view = []
    for view_index, index in enumerate(indices):
        original = frames[index]
        frame = resize_camera(original, render_width)
        mask_path = segmentation / segment_data["masks"][frame.frame_id]
        with Image.open(mask_path) as picture:
            if picture.size != (original.width, original.height):
                raise ValueError(f"{frame.frame_id} 掩码与源相机尺寸不一致")
            label = np.asarray(picture.resize((frame.width, frame.height), Image.Resampling.NEAREST)).astype(np.uint16)
        confidence_source = np.load(segmentation / segment_data["confidence"][frame.frame_id], allow_pickle=False).astype(np.float32)
        confidence = np.asarray(Image.fromarray(confidence_source).resize((frame.width, frame.height), Image.Resampling.BILINEAR))
        reliability = np.zeros(label.shape, dtype=np.float32)
        channels = np.full(label.shape, -1, dtype=np.int64)
        for channel, instance_id in enumerate(instance_values):
            region = label == instance_id
            distance = distance_transform_edt(region)
            # A soft three-pixel erosion suppresses uncertain mask borders.
            reliability[region] = np.clip((distance[region]-0.5)/3, 0, 1)*confidence[region]
            channels[region] = channel
        rendered, alphas, meta = renderer.render_geometry(frame, packed=False)
        gaussian_ids, pixel_ids, contribution = splat_contributions(renderer, frame, meta)
        with torch.inference_mode():
            if not len(gaussian_ids):
                continue
            pixel_channels = torch.tensor(channels.reshape(-1), device="cuda")[pixel_ids]
            pixel_reliability = torch.tensor(reliability.reshape(-1), device="cuda")[pixel_ids]
            if depth_reference == 'surface':
                surface_depth = pixel_surface_depths(pixel_ids,meta['depths'][0,gaussian_ids],contribution,frame.width*frame.height)
                depth = surface_depth[pixel_ids]
            else:
                depth = rendered[0,:,:,3].reshape(-1)[pixel_ids]
            alpha = alphas[0, :, :, 0].reshape(-1)[pixel_ids]
            projected = meta["means2d"][0, gaussian_ids]
            in_frame = (projected[:, 0] >= 0.5) & (projected[:, 0] < frame.width+0.5) & (projected[:, 1] >= 0.5) & (projected[:, 1] < frame.height+0.5)
            # Depth is a conservative consistency gate; T*alpha is actual visibility evidence.
            tolerance = 3*renderer.parameters["scales"][gaussian_ids].max(dim=1).values + 0.03*depth + 0.005
            depth_consistent = (meta["depths"][0, gaussian_ids]-depth).abs() <= tolerance
            valid = (pixel_channels >= 0) & (pixel_reliability > 0.25) & (contribution > 1e-4) & (depth > 0) & (alpha > 0.1) & in_frame & depth_consistent
            gaussian_ids, pixel_channels = gaussian_ids[valid], pixel_channels[valid]
            weights = (contribution*pixel_reliability)[valid]
            flat = torch.zeros(n*classes, device="cuda")
            flat.scatter_add_(0, gaussian_ids*classes+pixel_channels, weights)
            view_votes = flat.reshape(n, classes)
            current, current_support = weight_view_evidence(view_votes.cpu().numpy())
        votes += current
        support += current_support
        per_view.append({"frame_id": frame.frame_id, "contributing_gaussians": int((current.sum(axis=1) > 0).sum()), "weighted_pixels": int(valid.sum())})
        print(f"Fusion view {view_index+1}/{len(indices)}: {frame.frame_id}", flush=True)
        del rendered, alphas, meta, contribution, pixel_ids, gaussian_ids
    labels, confidence, views = finalize_votes(votes, support, instance_values, minimum_views=minimum_views)
    output = bundle / "semantics"
    initialize_scene(output)
    np.savez_compressed(output / "semantics.npz", gaussian_ids=renderer.model.gaussian_ids, instance_ids=labels, confidence=confidence, support_views=views, votes=votes, instance_values=instance_values)
    shutil.copy2(segmentation / "instances.json", output / "instances.json")
    result = {"schema_version": 1, "model_sha256": renderer.metadata["model_sha256"], "segmentation": str(segmentation), "visibility": "gsplat_pixel_intersections_front_to_back_T_alpha_plus_depth_gate", "boundary_weight": "soft_distance_transform_erosion", "minimum_support_views": minimum_views, "voting_views": [frames[index].frame_id for index in indices], "holdout_views": [frames[index].frame_id for index in holdout], "per_view": per_view, "gaussians_per_instance": {str(int(value)): int((labels == value).sum()) for value in np.unique(labels)}, "unknown_fraction": float((labels == UNKNOWN_ID).mean()), "confidence_type": "vote_fraction_not_calibrated_probability", "status": "fused_requires_holdout_and_click_review"}
    result['algorithm_version'] = 2
    result['support_rule'] = 'within_view_class_fraction_ge_0.5_and_T_alpha_reliability_mass_ge_0.001'
    result['minimum_view_evidence_mass'] = 0.001
    result['depth_gate_reference'] = depth_reference
    write_json(output / "fusion.json", result)
    return {key: result[key] for key in ["model_sha256", "gaussians_per_instance", "unknown_fraction", "status"]}
