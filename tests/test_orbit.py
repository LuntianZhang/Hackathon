import numpy as np
import pytest

from food3d.schemas import CameraFrame
from food3d.synthetic import look_at
from food3d.viewer.orbit import drag_camera


def camera():
    # Deliberate roll and off-center pivot: a world-up look_at would snap.
    pose = look_at(np.array([1., -2, 1.5]), np.zeros(3))
    angle = .6
    roll = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    pose[:3, :3] = pose[:3, :3] @ roll
    return CameraFrame('captured', '', 640, 360, [[600, 0, 319.5], [0, 600, 179.5], [0, 0, 1]], pose.tolist())


def project(camera, point):
    local = np.linalg.inv(camera.camera_to_world) @ np.r_[point, 1]
    pixel = np.asarray(camera.K) @ local[:3]
    return pixel[:2]/pixel[2]


def test_neutral_drag_preserves_captured_camera_and_roll():
    original = camera()
    changed = drag_camera(original, np.array([.1, .05, 0]), 0, 0, 1)
    np.testing.assert_allclose(changed.camera_to_world, original.camera_to_world, atol=1e-12)
    assert (changed.width, changed.height, changed.K) == (original.width, original.height, original.K)


def test_drag_keeps_pivot_at_same_pixel_and_scales_camera_distance():
    original = camera()
    pivot = np.array([.1, .05, 0])
    changed = drag_camera(original, pivot, 37, -22, 1.3)
    np.testing.assert_allclose(project(changed, pivot), project(original, pivot), atol=1e-9)
    old_distance = np.linalg.norm(np.asarray(original.camera_to_world)[:3, 3]-pivot)
    new_distance = np.linalg.norm(np.asarray(changed.camera_to_world)[:3, 3]-pivot)
    assert new_distance == pytest.approx(old_distance*1.3)
    changed.validate()


def test_drag_limits_extreme_zoom_and_rejects_nonfinite_values():
    pivot = np.zeros(3)
    far = drag_camera(camera(), pivot, 5000, 90, 100)
    far.validate()
    with pytest.raises(ValueError):
        drag_camera(camera(), pivot, float('nan'), 0, 1)
    with pytest.raises(ValueError):
        drag_camera(camera(), pivot, 0, 0, 0)
