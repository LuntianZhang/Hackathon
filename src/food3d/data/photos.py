"""Ordered photo import with EXIF orientation, quality reports and provenance."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ..io import initialize_scene, write_json
from ..visuals import contact_sheet
from .frames import image_metrics


def import_photos(source: Path, scene: Path, options: dict, source_kind: str = "photos") -> dict:
    source, scene = source.resolve(), scene.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"照片目录不存在：{source}")
    if source == scene or source in scene.parents or scene in source.parents:
        raise ValueError("照片源目录和场景输出目录不能互相包含")
    def natural_key(path: Path):
        return tuple((1, int(part)) if part.isdigit() else (0, part.casefold()) for part in re.split(r"(\d+)", path.name))
    files = sorted((path for path in source.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}), key=natural_key)
    if not files:
        raise ValueError("目录中没有支持的照片")
    maximum, edge = int(options.get("max_frames", 120)), int(options.get("max_long_edge", 960))
    min_sharpness, min_difference = float(options.get("min_sharpness", 12)), float(options.get("min_difference", 0))
    quality = int(options.get("jpeg_quality", 95))
    if maximum < 2 or edge < 16 or min_sharpness < 0 or min_difference < 0 or not 1 <= quality <= 100:
        raise ValueError("照片导入参数无效")
    selected = set(np.linspace(0, len(files)-1, min(maximum, len(files)), dtype=int).tolist())
    initialize_scene(scene)
    (scene / "frames").mkdir()
    previous = None
    manifest, thumbnails = [], []
    for index, path in enumerate(files):
        frame_id = f"{index:06d}"
        record = {"frame_id": frame_id, "sample_index": index, "source_filename": path.name, "source_path": str(path), "timestamp_seconds": None, "image_path": None, "accepted": False}
        if index not in selected:
            record["rejection_reason"] = "uniform_subsampling"
            manifest.append(record)
            continue
        with Image.open(path) as original:
            original_size = original.size
            oriented = ImageOps.exif_transpose(original)
            oriented_size = oriented.size
            image = oriented.convert("RGB")
            image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        sharpness, difference, thumbnail = image_metrics(image, previous)
        reason = "blur" if sharpness < min_sharpness else "duplicate" if difference is not None and difference < min_difference else None
        if reason is None:
            output = scene / "frames" / f"{frame_id}.jpg"
            image.save(output, quality=quality)
            record["image_path"] = output.relative_to(scene).as_posix()
            record["accepted"] = True
            previous = thumbnail
        record.update({"width": image.width, "height": image.height, "original_size": list(original_size), "oriented_size": list(oriented_size), "scale_xy": [image.width/oriented_size[0], image.height/oriented_size[1]], "sharpness": sharpness, "difference_from_last_accepted": difference, "rejection_reason": reason})
        preview = image.copy()
        preview.thumbnail((248, 188))
        thumbnails.append((f"{frame_id} {reason or 'accepted'} s={sharpness:.1f}", preview))
        manifest.append(record)
    result = {"schema_version": 1, "source_type": "provided_rendered_images" if source_kind == "rendered" else "provided_photos", "source_directory": str(source), "ordering": "natural_filename_order", "timestamp_method": "not_available_use_order_only", "options": options, "frames": manifest}
    write_json(scene / "frame_manifest.json", result)
    contact_sheet(thumbnails, scene / "debug" / "frame_sheet.png")
    summary = {"source": result["source_type"], "stage": "photo_import", "source_images": len(files), "sampled_images": len(selected), "accepted_frames": sum(frame["accepted"] for frame in manifest), "reconstruction_verified": False, "semantics_verified": False}
    write_json(scene / "validation.json", summary)
    if summary["accepted_frames"] < 2:
        raise RuntimeError("有效照片少于两张，请检查质量筛选阈值")
    return summary
