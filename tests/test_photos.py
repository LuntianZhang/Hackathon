import numpy as np
from PIL import Image

from food3d.data.photos import import_photos
from food3d.io import read_json


def test_photo_import_preserves_order_and_marks_missing_timestamps(tmp_path):
    source = tmp_path / "photos"
    source.mkdir()
    random = np.random.default_rng(1)
    for name in ["10.jpg", "2.jpg", "1.jpg"]:
        Image.fromarray(random.integers(0, 256, (80, 120, 3), dtype=np.uint8)).save(source / name)
    scene = tmp_path / "scene"
    result = import_photos(source, scene, {"max_long_edge": 96, "min_sharpness": 0, "min_difference": 0}, "rendered")
    manifest = read_json(scene / "frame_manifest.json")
    assert result["accepted_frames"] == 3
    assert manifest["source_type"] == "provided_rendered_images"
    assert [frame["source_filename"] for frame in manifest["frames"]] == ["1.jpg", "2.jpg", "10.jpg"]
    assert all(frame["timestamp_seconds"] is None for frame in manifest["frames"])
    for frame in manifest["frames"]:
        with Image.open(scene / frame["image_path"]) as image:
            assert image.size == (96, 64)
