"""Extract timestamped video samples, then reject blurry or duplicate images."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from ..io import initialize_scene, write_json
from ..paths import ProjectPaths
from ..visuals import contact_sheet


def image_metrics(image: Image.Image, previous: np.ndarray | None) -> tuple[float, float | None, np.ndarray]:
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    if min(gray.shape) < 3:
        raise ValueError("抽帧图像尺寸过小")
    laplacian = -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    thumbnail = np.asarray(image.convert("L").resize((64, 64)), dtype=np.float32)
    difference = None if previous is None else float(np.abs(thumbnail - previous).mean())
    return float(laplacian.var()), difference, thumbnail


def extract_video(video: Path, scene: Path, options: dict) -> dict:
    video = video.resolve()
    scene = scene.resolve()
    if not video.is_file():
        raise FileNotFoundError(f"找不到视频：{video}")
    tools = ProjectPaths().tools_root / 'ffmpeg/bin'
    ffmpeg = shutil.which(str(options.get('ffmpeg', tools/'ffmpeg.exe')))
    ffprobe = shutil.which(str(options.get('ffprobe', tools/'ffprobe.exe')))
    if not ffmpeg or not ffprobe:
        raise RuntimeError("需要将现代 FFmpeg 和 FFprobe 加入 PATH")
    fps = float(options.get("fps", 3))
    limit = int(options.get("max_frames", 120))
    edge = int(options.get("max_long_edge", 960))
    quality = int(options.get("jpeg_quality", 95))
    min_sharpness = float(options.get("min_sharpness", 12))
    min_difference = float(options.get("min_difference", 2))
    if fps <= 0 or limit <= 0 or edge < 16 or not 1 <= quality <= 100 or min_sharpness < 0 or min_difference < 0:
        raise ValueError("抽帧参数无效")
    # FFprobe JSON is UTF-8 even when Windows' locale encoding is GBK.
    probe = subprocess.run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)], capture_output=True, text=True, encoding="utf-8", check=True)
    import json
    metadata = json.loads(probe.stdout)
    if not any(stream.get("codec_type") == "video" for stream in metadata.get("streams", [])):
        raise ValueError("输入没有视频流")
    initialize_scene(scene)
    samples = scene / "samples"
    images = scene / "frames"
    samples.mkdir()
    images.mkdir()
    # select preserves original sample PTS; fps would create synthetic timestamps.
    filter_chain = (
        f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,{1/fps:.12f})',"
        f"scale=w='min(iw,{edge})':h='min(ih,{edge})':force_original_aspect_ratio=decrease,showinfo"
    )
    command = [ffmpeg, "-hide_banner", "-nostdin", "-i", str(video), "-map", "0:v:0", "-an", "-vf", filter_chain, "-vsync", "vfr", "-frames:v", str(limit), "-start_number", "0", "-q:v", "2", str(samples / "%06d.jpg")]
    log = scene / "logs" / "ffmpeg.log"
    log.parent.mkdir()
    with log.open("w", encoding="utf-8") as stream:
        stream.write("COMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        stream.flush()
        process = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=stream, check=False)
    if process.returncode:
        raise RuntimeError(f"视频解码失败，请查看 {log}")
    pts = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.search(r"\bn:\s*(\d+).*?\bpts_time:\s*([-+0-9.eE]+)", line)
        if match:
            pts[int(match.group(1))] = float(match.group(2))
    files = sorted(samples.glob("*.jpg"))
    if not files:
        raise RuntimeError("FFmpeg 没有输出帧")
    first_pts = pts.get(0)
    if first_pts is None:
        raise RuntimeError(f"无法读取真实抽帧时间戳，请检查 FFmpeg 版本和 {log}")
    previous = None
    manifest = []
    thumbnails = []
    for path in files:
        sample_index = int(path.stem)
        if sample_index not in pts:
            raise RuntimeError(f"帧 {sample_index} 缺少时间戳，拒绝估算")
        with Image.open(path) as source:
            image = source.convert("RGB")
        sharpness, difference, thumbnail = image_metrics(image, previous)
        reason = "blur" if sharpness < min_sharpness else "duplicate" if difference is not None and difference < min_difference else None
        accepted = reason is None
        target = images / f"{sample_index:06d}.jpg"
        if accepted:
            image.save(target, quality=quality)
            previous = thumbnail
        preview = image.copy()
        preview.thumbnail((248, 188))
        thumbnails.append((f"{path.stem} {reason or 'accepted'} s={sharpness:.1f}", preview))
        manifest.append({
            "frame_id": path.stem, "sample_index": sample_index,
            "sample_path": path.relative_to(scene).as_posix(),
            "image_path": target.relative_to(scene).as_posix() if accepted else None,
            "video_pts_seconds": pts[sample_index],
            "timestamp_seconds": pts[sample_index] - first_pts,
            "width": image.width, "height": image.height,
            "sharpness": sharpness, "difference_from_last_accepted": difference,
            "accepted": accepted, "rejection_reason": reason,
        })
    result = {"schema_version": 1, "source_video": str(video), "timestamp_origin_video_pts": first_pts, "timestamp_method": "ffmpeg_select_showinfo", "options": options, "frames": manifest}
    write_json(scene / "video_metadata.json", metadata)
    write_json(scene / "frame_manifest.json", result)
    contact_sheet(thumbnails, scene / "debug" / "frame_sheet.png")
    write_json(scene / "validation.json", {"source": "real_video", "stage": "frames", "sampled_frames": len(files), "accepted_frames": sum(frame["accepted"] for frame in manifest), "reconstruction_verified": False, "semantics_verified": False})
    if not any(frame["accepted"] for frame in manifest):
        raise RuntimeError("全部帧被质量筛选排除，请检查缩略图与阈值")
    return result

