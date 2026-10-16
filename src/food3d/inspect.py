"""Contact sheets and sparse-point projections for manual camera review."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data.cameras import project_points
from .io import write_json
from .schemas import load_scene
from .visuals import contact_sheet


def inspect_scene(scene: Path, max_points: int = 1200) -> dict:
    scene = scene.resolve()
    if max_points < 1:
        raise ValueError("max_points 必须为正数")
    metadata, frames = load_scene(scene / "scene.json")
    debug = scene / "debug"
    debug.mkdir(exist_ok=True)
    cloud_path = scene / "sparse_points.npz"
    if cloud_path.exists():
        with np.load(cloud_path, allow_pickle=False) as cloud:
            points = cloud["positions"]
        if len(points) > max_points:
            points = points[np.linspace(0, len(points) - 1, max_points, dtype=int)]
    else:
        points = np.empty((0, 3))
    originals, overlays, summary = [], [], []
    for frame in frames:
        with Image.open(scene / frame.image_path) as source:
            image = source.convert("RGB")
        if image.size != (frame.width, frame.height):
            raise ValueError(f"{frame.frame_id} 图像尺寸与相机不一致")
        pixels, depth, in_frame = project_points(points, frame)
        visible = in_frame.copy()
        # Ground-truth depth is only available for synthetic scenes at this stage.
        if frame.depth_path:
            surface = np.load(scene / frame.depth_path, allow_pickle=False)
            if surface.shape != (frame.height, frame.width):
                raise ValueError(f"{frame.frame_id} 深度尺寸与相机不一致")
            indices = np.flatnonzero(in_frame)
            xy = np.rint(pixels[indices]).astype(int)
            xy[:, 0] = np.clip(xy[:, 0], 0, frame.width - 1)
            xy[:, 1] = np.clip(xy[:, 1], 0, frame.height - 1)
            observed = surface[xy[:, 1], xy[:, 0]]
            visible[indices] = (observed > 0) & (np.abs(depth[indices] - observed) <= np.maximum(0.005, observed * 0.015))
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay)
        for x, y in pixels[visible]:
            draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(30, 255, 80))
        overlay.save(debug / f"projection_{frame.frame_id}.png")
        image.thumbnail((248, 188))
        overlay.thumbnail((248, 188))
        originals.append((frame.frame_id, image))
        overlays.append((frame.frame_id, overlay))
        if frame.mask_path:
            with Image.open(scene / frame.mask_path) as source:
                mask = np.asarray(source)
            if mask.shape != (frame.height, frame.width):
                raise ValueError(f"{frame.frame_id} 掩码尺寸与相机不一致")
            palette = np.array([[70, 70, 70], [220, 50, 40], [240, 205, 30], [185, 120, 55]], dtype=np.uint8)
            colored = np.full((*mask.shape, 3), 255, dtype=np.uint8)
            valid = mask < len(palette)
            colored[valid] = palette[mask[valid]]
            Image.fromarray(colored).save(debug / f"instances_{frame.frame_id}.png")
        summary.append({"frame_id": frame.frame_id, "projected_in_frame": int(in_frame.sum()), "displayed_points": int(visible.sum()), "visibility_method": "approximate_ground_truth_depth_gate" if frame.depth_path else "in_frame_only_no_occlusion_check"})
    contact_sheet(originals, debug / "contact_sheet.png")
    contact_sheet(overlays, debug / "projection_sheet.png")
    result = {"source": metadata["source"], "cameras": len(frames), "sampled_points": len(points), "frames": summary, "status": "diagnostics_generated_requires_manual_review", "notice": "图像生成和文件检查不代表三维重建或语义融合已验证。"}
    write_json(debug / "inspection.json", result)
    return result

