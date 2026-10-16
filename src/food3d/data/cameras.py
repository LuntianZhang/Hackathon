from __future__ import annotations

from pathlib import Path

import numpy as np

from ..io import write_json
from ..schemas import CameraFrame


def quaternion_to_rotation(q: np.ndarray) -> np.ndarray:
    """COLMAP quaternion order is w, x, y, z."""
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("无效四元数")
    w, x, y, z = q / norm
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def project_points(points: np.ndarray, camera: CameraFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera.validate()
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("三维点必须为 N x 3")
    pose = np.asarray(camera.camera_to_world)
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    depth = local[:, 2]
    projected = local @ np.asarray(camera.K).T
    pixels = np.full((len(points), 2), np.nan)
    front = np.isfinite(local).all(axis=1) & (depth > 1e-8)
    pixels[front] = projected[front, :2] / depth[front, None]
    visible = front & (pixels[:, 0] >= 0) & (pixels[:, 0] < camera.width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < camera.height)
    return pixels, depth, visible


def unproject_pixels(pixels: np.ndarray, depth: np.ndarray, camera: CameraFrame) -> np.ndarray:
    camera.validate()
    pixels = np.asarray(pixels, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or depth.shape != (len(pixels),):
        raise ValueError("像素必须为 N x 2，深度必须为 N")
    if not np.isfinite(pixels).all() or not np.isfinite(depth).all() or (depth <= 0).any():
        raise ValueError("反投影需要有限像素和正深度")
    rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(np.asarray(camera.K)).T
    local = rays * depth[:, None]
    pose = np.asarray(camera.camera_to_world)
    return local @ pose[:3, :3].T + pose[:3, 3]


def export_nerfstudio_transforms(path: Path, frames: list[CameraFrame], sparse_point_path: str | None = None) -> None:
    converted = []
    for frame in frames:
        frame.validate()
        k = np.asarray(frame.K)
        pose = np.asarray(frame.camera_to_world) @ np.diag([1, -1, -1, 1])
        converted.append({
            "file_path": frame.image_path,
            "w": frame.width, "h": frame.height,
            "fl_x": float(k[0, 0]), "fl_y": float(k[1, 1]),
            # COLMAP/OpenCV pixel center -> Nerfstudio's half-offset center.
            "cx": float(k[0, 2] + 0.5), "cy": float(k[1, 2] + 0.5),
            "transform_matrix": pose.tolist(),
        })
    metadata = {
        "camera_model": "OPENCV", "k1": 0.0, "k2": 0.0, "p1": 0.0, "p2": 0.0,
        # Saved poses and PLY already share the canonical COLMAP world.
        # Explicit identity prevents Nerfstudio's legacy COLMAP-directory heuristic.
        "applied_transform": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        "frames": converted,
    }
    if sparse_point_path:
        metadata["ply_file_path"] = sparse_point_path
    write_json(path, metadata)

