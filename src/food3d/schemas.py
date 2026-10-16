"""Canonical on-disk camera contract: OpenCV camera axes, camera-to-world poses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .io import read_json, write_json

BACKGROUND_ID = 0
UNKNOWN_ID = 65535


@dataclass
class CameraFrame:
    frame_id: str
    image_path: str
    width: int
    height: int
    K: list[list[float]]
    camera_to_world: list[list[float]]
    source_frame_id: str | None = None
    timestamp_seconds: float | None = None
    colmap_image_id: int | None = None
    colmap_camera_id: int | None = None
    mask_path: str | None = None
    depth_path: str | None = None

    def validate(self) -> None:
        k = np.asarray(self.K, dtype=np.float64)
        pose = np.asarray(self.camera_to_world, dtype=np.float64)
        if self.width < 1 or self.height < 1:
            raise ValueError("图像尺寸必须为正数")
        if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError("K 必须是有限的 3x3 内参矩阵，焦距为正数")
        if not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("K 的末行必须为 [0, 0, 1]")
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError("位姿必须为有效的 4x4 齐次矩阵")
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
            raise ValueError("位姿旋转必须为正交且行列式为 +1")
        if self.timestamp_seconds is not None and (not np.isfinite(self.timestamp_seconds) or self.timestamp_seconds < 0):
            raise ValueError("时间戳必须有限且非负")


def save_scene(path: Path, frames: list[CameraFrame], *, source: str, world_units: str, metadata: dict | None = None) -> None:
    if not frames:
        raise ValueError("场景至少需要一个有效相机")
    for frame in frames:
        frame.validate()
    if len({frame.frame_id for frame in frames}) != len(frames):
        raise ValueError("frame_id 必须唯一")
    write_json(path, {
        "schema_version": 1,
        "source": source,
        "camera_convention": "opencv",
        "pose_convention": "camera_to_world",
        "pixel_convention": "integer_coordinates_at_pixel_centers",
        "depth_convention": "camera_z",
        "world_units": world_units,
        "images_undistorted": True,
        "frames": [asdict(frame) for frame in frames],
        "metadata": metadata or {},
    })


def load_scene(path: Path) -> tuple[dict, list[CameraFrame]]:
    data = read_json(path)
    if (data.get("schema_version"), data.get("camera_convention"), data.get("pose_convention")) != (1, "opencv", "camera_to_world"):
        raise ValueError("不支持的场景数据版本或相机坐标约定")
    if not data.get("images_undistorted"):
        raise ValueError("当前接口仅接受去畸变图像")
    frames = [CameraFrame(**frame) for frame in data["frames"]]
    if not frames or len({frame.frame_id for frame in frames}) != len(frames):
        raise ValueError("相机列表不能为空且 frame_id 必须唯一")
    for frame in frames:
        frame.validate()
    return data, frames

