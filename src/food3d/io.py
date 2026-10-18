"""Shared configuration, JSON and external program helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import yaml
from .paths import ProjectPaths


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("配置必须是 schema_version: 1 的 YAML 对象")
    colmap = value.get('colmap', {})
    if colmap.get('reuse_database'):
        colmap['reuse_database'] = str(ProjectPaths().resolve(colmap['reuse_database']))
    frames = value.get('frames', {})
    for name in ('ffmpeg', 'ffprobe'):
        executable = frames.get(name)
        if executable and Path(executable).parent != Path('.'):
            frames[name] = str(ProjectPaths().resolve(executable))
    return value


def run_logged(command: list[str], log: Path) -> None:
    """No shell interpolation; keep framework diagnostics in a readable log."""
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("\nCOMMAND " + json.dumps(command, ensure_ascii=False) + "\n")
        stream.flush()
        # Python 3.10 site.py decodes Windows .pth files using the locale;
        # editable installs in a Chinese path may be GBK encoded. Keep that
        # locale while making redirected framework logs explicitly UTF-8.
        environment = ProjectPaths().environment()
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False, env=environment)
    if result.returncode:
        raise RuntimeError(f"外部程序退出码 {result.returncode}，请查看 {log}")


def initialize_scene(path: Path) -> None:
    # Never mix artifacts from two independent runs or overwrite a video dataset.
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"输出目录必须不存在或为空：{path}")
    path.mkdir(parents=True, exist_ok=True)


def write_point_cloud_ply(path: Path, positions: Any, colors: Any) -> None:
    """Small ASCII PLY export for Nerfstudio's sparse-point initialization."""
    import numpy as np
    positions = np.asarray(positions, dtype=float)
    colors = np.asarray(colors)
    if positions.ndim != 2 or positions.shape[1] != 3 or colors.shape != positions.shape:
        raise ValueError("点云位置与颜色都必须是 N x 3")
    if not np.isfinite(positions).all() or not np.isfinite(colors).all():
        raise ValueError("点云必须为有限数值")
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(positions)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(positions, colors):
            stream.write(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} {color[0]} {color[1]} {color[2]}\n")

