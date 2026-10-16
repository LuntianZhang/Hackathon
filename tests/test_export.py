import numpy as np

from food3d.reconstruction.export import transform_camera, camera_world_transform
from food3d.viewer.renderer import resize_camera
from food3d.schemas import CameraFrame


def test_training_scale_applies_to_translation_but_preserves_rotation():
    pose = np.eye(4)
    pose[:3, 3] = [1, 2, 3]
    transform = np.array([[0, -1, 0, 4], [1, 0, 0, 5], [0, 0, 1, 6]])
    result = transform_camera(pose, transform, 2)
    np.testing.assert_allclose(result[:3, 3], [4, 12, 18])
    np.testing.assert_allclose(result[:3, :3].T @ result[:3, :3], np.eye(3))


def test_camera_resize_preserves_half_pixel_geometry():
    camera = CameraFrame("a", "a.jpg", 640, 480, [[500, 0, 319.5], [0, 500, 239.5], [0, 0, 1]], np.eye(4).tolist())
    smaller = resize_camera(camera, 320)
    np.testing.assert_allclose(smaller.K, [[250, 0, 159.5], [0, 250, 119.5], [0, 0, 1]])
    assert smaller.height == 240


def test_legacy_colmap_transform_is_removed_from_saved_camera_map():
    canonical = np.eye(4)
    canonical[:3, 3] = [1, 2, 3]
    legacy = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    reported = {"transform": (canonical @ legacy)[:3].tolist(), "scale": 2}
    corrected, scale = camera_world_transform(reported, {"transforms": {}, "legacy_colmap_layout_detected": True})
    np.testing.assert_allclose(corrected, canonical)
    assert scale == 2
