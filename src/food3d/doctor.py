"""Read-only environment report. Importing torch is optional and isolated."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys


def report() -> dict:
    packages = {}
    for name in ["numpy", "Pillow", "PyYAML", "torch", "torchvision", "nerfstudio", "gsplat", "SAM-2", "open_clip_torch", "gradio", "pytest"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    tools = {name: shutil.which(name) for name in ["ffmpeg", "ffprobe", "colmap", "nvcc", "nvidia-smi", "ns-train", "ns-process-data"]}
    result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": packages, "tools": tools, "validation_status": "environment_only"}
    if tools["nvidia-smi"]:
        try:
            process = subprocess.run([tools["nvidia-smi"], "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
            result["gpu"] = {"returncode": process.returncode, "output": process.stdout.strip(), "error": process.stderr.strip()}
        except (OSError, subprocess.TimeoutExpired) as error:
            result["gpu"] = {"error": str(error)}
    if packages["torch"]:
        code = "import json, torch; from torch.utils.cpp_extension import CUDA_HOME; print(json.dumps({'version':torch.__version__,'runtime_cuda':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'cuda_home':CUDA_HOME}))"
        try:
            process = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
            result["torch"] = json.loads(process.stdout) if process.returncode == 0 else {"error": process.stderr[-4000:]}
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            result["torch"] = {"error": str(error)}
    return result

