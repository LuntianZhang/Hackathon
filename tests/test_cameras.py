"""Geometric acceptance checks; authored but not executed during initial setup."""

import numpy as np
import pytest

from food3d.data.cameras import export_nerfstudio_transforms, project_points, unproject_pixels
from food3d.data.colmap import read_cameras, read_images
from food3d.io import read_json
from food3d.schemas import CameraFrame


def test_projection_round_trip_with_rotated_translated_camera():
    pose = np.array([[0, 0, 1, 2], [0, 1, 0, -1], [-1, 0, 0, 3], [0, 0, 0, 1]], dtype=float)
    camera = CameraFrame("one", "images/one.png", 640, 480, [[500, 0, 319.5], [0, 510, 239.5], [0, 0, 1]], pose.tolist())
    pixels = np.array([[100, 150], [319.5, 239.5], [500, 300]])
    depth = np.array([1.2, 2.5, 4.0])
    points = unproject_pixels(pixels, depth, camera)
    actual_pixels, actual_depth, in_frame = project_points(points, camera)
    np.testing.assert_allclose(actual_pixels, pixels, atol=1e-9)
    np.testing.assert_allclose(actual_depth, depth, atol=1e-9)
    assert in_frame.all()
    behind = pose[:3, 3] - pose[:3, 2]
    _, _, visible = project_points(behind[None, :], camera)
    assert not visible[0]


def test_colmap_to_nerfstudio_preserves_physical_rays(tmp_path):
    cameras_txt = tmp_path / "cameras.txt"
    cameras_txt.write_text("# cameras\n1 PINHOLE 640 480 500 510 320 240\n", encoding="utf-8")
    width, height, k = read_cameras(cameras_txt)[1]
    camera = CameraFrame("one", "images/one.png", width, height, k.tolist(), np.eye(4).tolist())
    path = tmp_path / "transforms.json"
    export_nerfstudio_transforms(path, [camera])
    frame = read_json(path)["frames"][0]
    # Center of top-left pixel is integer in canonical data and +0.5 in Nerfstudio.
    ray_cv = np.linalg.inv(k) @ np.array([0, 0, 1])
    ray_gl = np.array([(0.5 - frame["cx"]) / frame["fl_x"], -(0.5 - frame["cy"]) / frame["fl_y"], -1])
    np.testing.assert_allclose(np.asarray(frame["transform_matrix"])[:3, :3] @ ray_gl, ray_cv)


def test_colmap_empty_observation_lines_and_names_with_spaces(tmp_path):
    path = tmp_path / "images.txt"
    path.write_text("# images\n1 1 0 0 0 0 0 0 1 first image.jpg\n\n2 1 0 0 0 1 0 0 1 second.jpg\n10 20 -1\n", encoding="utf-8")
    images = read_images(path)
    assert [image["name"] for image in images] == ["first image.jpg", "second.jpg"]


def test_reject_non_rotation_pose():
    pose = np.eye(4)
    pose[0, 0] = 2
    camera = CameraFrame("bad", "bad.png", 640, 480, [[500, 0, 320], [0, 500, 240], [0, 0, 1]], pose.tolist())
    with pytest.raises(ValueError, match="正交"):
        camera.validate()

