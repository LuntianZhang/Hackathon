import numpy as np
import pytest
from PIL import Image

from food3d.data.cameras import project_points, unproject_pixels
from food3d.schemas import load_scene
from food3d.synthetic import generate_scene


def test_synthetic_depth_mask_and_camera_alignment(tmp_path):
    scene = tmp_path / "synthetic"
    generate_scene(scene, {"views": 4, "width": 96, "height": 72, "seed": 7})
    metadata, frames = load_scene(scene / "scene.json")
    assert metadata["source"] == "synthetic"
    seen = set()
    for frame in frames:
        with Image.open(scene / frame.mask_path) as image:
            labels = np.asarray(image)
        depth = np.load(scene / frame.depth_path)
        assert labels.dtype in (np.dtype("uint16"), np.dtype("int32"))
        assert set(np.unique(labels)) <= {0, 1, 2, 3}
        seen.update(np.unique(labels).tolist())
        assert (depth[labels > 0] > 0).all()
        valid = depth > 0
        valid[[0, -1], :] = False
        valid[:, [0, -1]] = False
        y, x = np.nonzero(valid)
        pixels = np.column_stack([x[::31], y[::31]])
        z = depth[y[::31], x[::31]]
        points = unproject_pixels(pixels, z, frame)
        actual, actual_z, visible = project_points(points, frame)
        np.testing.assert_allclose(actual, pixels, atol=1e-5)
        np.testing.assert_allclose(actual_z, z, atol=1e-6)
        assert visible.all()
    assert seen == {0, 1, 2, 3}


def test_generator_refuses_to_overwrite_existing_scene(tmp_path):
    scene = tmp_path / "existing"
    scene.mkdir()
    (scene / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        generate_scene(scene, {})
    assert (scene / "keep.txt").read_text(encoding="utf-8") == "keep"

