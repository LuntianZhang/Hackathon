"""Camera-relative orbit around a stable 3D pivot; preserve captured roll."""
from dataclasses import replace

import numpy as np

from ..schemas import CameraFrame


def axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*cross + (1-np.cos(angle))*(cross @ cross)


def drag_camera(base: CameraFrame, pivot: np.ndarray, yaw: float, pitch: float, zoom: float) -> CameraFrame:
    base.validate()
    pivot = np.asarray(pivot, dtype=float)
    if pivot.shape != (3,) or not np.isfinite(pivot).all() or not np.isfinite([yaw, pitch, zoom]).all():
        raise ValueError("拖动相机参数必须为有限数值")
    if zoom <= 0:
        raise ValueError("相机距离倍率必须大于零")
    pose = np.asarray(base.camera_to_world, dtype=float)
    # OpenCV local y points down. Yaw rotates around the initial camera's up;
    # pitch rotates around its yaw-rotated right. Both position and orientation
    # rotate together, preserving the pivot's pixel projection and camera roll.
    horizontal = axis_rotation(-pose[:3, 1], np.deg2rad(yaw % 360))
    vertical = axis_rotation(horizontal @ pose[:3, 0], np.deg2rad(np.clip(pitch, -75, 75)))
    rotation = vertical @ horizontal
    new_pose = np.eye(4)
    new_pose[:3, :3] = rotation @ pose[:3, :3]
    new_pose[:3, 3] = pivot + rotation @ (pose[:3, 3]-pivot)*np.clip(zoom, .5, 2.5)
    camera = replace(base, frame_id="drag_orbit", image_path="", camera_to_world=new_pose.tolist())
    camera.validate()
    return camera
