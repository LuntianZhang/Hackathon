from __future__ import annotations

import subprocess
from pathlib import Path

from ..io import run_logged, write_json


def train(scene: Path, output: Path, python: Path, iterations: int = 3000, config: Path | None = None) -> dict:
    scene, output, python = scene.resolve(), output.resolve(), python.absolute()
    if not python.is_file():
        raise FileNotFoundError(f"GPU 环境 Python 不存在：{python}")
    if not (scene / "transforms.json").is_file() or iterations < 1:
        raise ValueError("需要已完成的相机数据和正训练步数")
    output.mkdir(parents=True, exist_ok=True)
    before = set(output.rglob("*.ckpt"))
    command = [str(python), "-m", "food3d.reconstruction.train_worker", "--scene", str(scene), "--output", str(output), "--iterations", str(iterations)]
    if config is not None:
        command.extend(['--config',str(config.resolve())])
    run_logged(command, output / "training.log")
    checkpoints = set(output.rglob("*.ckpt")) - before
    if not checkpoints:
        raise RuntimeError("训练未产生新的检查点，请检查 training.log")
    checkpoint = max(checkpoints, key=lambda path: path.stat().st_mtime)
    run_dir = checkpoint.parent.parent
    result = {"source_scene": str(scene), "checkpoint": str(checkpoint), "config": str(run_dir / "config.yml"), "dataparser_transform": str(run_dir / "dataparser_transforms.json"), "iterations": iterations, "status": "training_complete_requires_render_review"}
    write_json(output / "latest_training.json", result)
    return result
