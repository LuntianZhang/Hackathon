import numpy as np
import json
from types import SimpleNamespace
from PIL import Image

from food3d.data.frames import image_metrics


def test_quality_metrics_separate_flat_texture_and_duplicate():
    uniform = Image.fromarray(np.full((64, 64), 128, dtype=np.uint8))
    sharpness, difference, thumbnail = image_metrics(uniform, None)
    assert sharpness == 0
    assert difference is None
    _, difference, _ = image_metrics(uniform, thumbnail)
    assert difference == 0
    y, x = np.mgrid[:64, :64]
    checkerboard = Image.fromarray((((x + y) % 2) * 255).astype(np.uint8))
    sharpness, difference, _ = image_metrics(checkerboard, thumbnail)
    assert sharpness > 1000
    assert difference > 100


def test_video_probe_utf8_metadata_and_configured_tools(tmp_path, monkeypatch):
    from food3d.data.frames import extract_video
    from food3d.io import read_json
    from pathlib import Path
    video = tmp_path / "水果.mp4"
    video.write_bytes(b"placeholder")
    scene = tmp_path / "scene"
    monkeypatch.setattr("food3d.data.frames.shutil.which", lambda name: name)
    metadata = {"streams": [{"codec_type": "video"}], "format": {"tags": {"title": "水果🍌"}}}
    def run(command, **kwargs):
        if command[0] == "modern-ffprobe":
            raw = json.dumps(metadata, ensure_ascii=False).encode("utf-8")
            return SimpleNamespace(returncode=0, stdout=raw.decode(kwargs.get("encoding", "gbk")))
        assert command[0] == "modern-ffmpeg"
        Image.new("RGB", (32, 24), "red").save(Path(command[-1].replace("%06d", "000000")))
        kwargs["stderr"].write("[showinfo] n: 0 pts: 3 pts_time: 0.125\n")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr("food3d.data.frames.subprocess.run", run)
    result = extract_video(video, scene, {"ffmpeg": "modern-ffmpeg", "ffprobe": "modern-ffprobe", "min_sharpness": 0})
    assert read_json(scene / "video_metadata.json")["format"]["tags"]["title"] == "水果🍌"
    assert result["frames"][0]["video_pts_seconds"] == .125
    assert result["frames"][0]["timestamp_seconds"] == 0

