"""Start the validated local PanSense workflow with a generated host config."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from food3d.paths import ProjectPaths  # noqa: E402


def executable(name: str, configured: str) -> str:
    bundled = (PROJECT / configured).resolve()
    if bundled.is_file():
        return str(bundled)
    found = shutil.which(name)
    return str(Path(found).resolve()) if found else name


def main() -> int:
    parser = argparse.ArgumentParser(description="启动本机 PanSense 视频或手机扫描流程")
    parser.add_argument("--mobile", action="store_true", help="同时开启手机 HTTPS 拍摄页")
    parser.add_argument("--colmap", choices=("cpu", "gpu"), default="cpu", help="COLMAP 特征提取与匹配设备")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", default="data/pansense")
    parser.add_argument("--lan-host", help="手机可访问的电脑局域网地址")
    parser.add_argument("--check", action="store_true", help="只检查并打印运行配置，不启动服务")
    args = parser.parse_args()

    paths = ProjectPaths(PROJECT)
    recognition = paths.recognition_python
    if not recognition.is_file():
        parser.error(f"识别环境不存在：{recognition}；请先按 environments/README.md 安装")
    if not (paths.tools_root / "weights/sam2.1_hiera_small.pt").is_file():
        parser.error("缺少 SAM 2 权重；请先按 environments/README.md 安装")
    if not 1 <= args.port <= 65535:
        parser.error("端口需在 1–65535 之间")

    scene = yaml.safe_load((PROJECT / "configs/fruit2_video.yaml").read_text(encoding="utf-8"))
    app = yaml.safe_load((PROJECT / "configs/pansense.yaml").read_text(encoding="utf-8"))
    scene["frames"]["ffmpeg"] = executable("ffmpeg", scene["frames"]["ffmpeg"])
    scene["frames"]["ffprobe"] = executable("ffprobe", scene["frames"]["ffprobe"])
    scene["frames"]["fps"] = 3
    scene["colmap"]["use_gpu"] = args.colmap == "gpu"
    scene["colmap"]["matching_use_gpu"] = args.colmap == "gpu"
    scene["reconstruction"]["stop_split_at"] = 12000
    app["iterations"] = 2000
    app["stop_split_at"] = 12000

    profile = paths.data_root / "runtime_profiles" / args.colmap
    profile.mkdir(parents=True, exist_ok=True)
    scene_path = profile / "scene.yaml"
    app_path = profile / "app.yaml"
    app["scene_config"] = str(scene_path)
    scene_path.write_text(yaml.safe_dump(scene, allow_unicode=True, sort_keys=False), encoding="utf-8")
    app_path.write_text(yaml.safe_dump(app, allow_unicode=True, sort_keys=False), encoding="utf-8")

    data_dir = paths.resolve(args.data_dir)
    command = [sys.executable, "-m", "pansense", "--port", str(args.port), "--data-dir", str(data_dir)]
    if args.mobile:
        command.append("--mobile")
        if args.lan_host:
            command.extend(("--lan-host", args.lan_host))
    environment = dict(os.environ)
    environment["PANSENSE_CONFIG"] = str(app_path)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(PROJECT / "src"), environment.get("PYTHONPATH", ""))))
    print(f"PanSense：http://127.0.0.1:{args.port}/", flush=True)
    print(f"COLMAP：{args.colmap.upper()}；餐食数据：{data_dir}", flush=True)
    if args.mobile:
        print("手机拍摄：启动后用电脑页面中的手机扫描二维码连接", flush=True)
    if args.check:
        print(f"运行配置：{app_path}", flush=True)
        print(f"FFmpeg：{scene['frames']['ffmpeg']}", flush=True)
        return 0
    if os.name == "nt":
        return subprocess.call(command, cwd=PROJECT, env=environment)
    os.chdir(PROJECT)
    os.execvpe(sys.executable, command, environment)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
