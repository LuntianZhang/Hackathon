"""Runs the official Splatfacto configuration in its dedicated GPU environment."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from food3d.paths import ProjectPaths, resolve_arguments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--config", type=Path)
    args = resolve_arguments(parser.parse_args())
    if args.iterations < 1 or not (args.scene / "transforms.json").is_file():
        raise ValueError("需要有效的 Nerfstudio 输入和正训练步数")
    cache = ProjectPaths().cache_root / 'models'
    os.environ.setdefault("TORCH_HOME", str(cache / "torch"))
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Splatfacto 需要 CUDA GPU")
    from nerfstudio.configs.method_configs import method_configs
    from nerfstudio.scripts.train import train_loop
    config = copy.deepcopy(method_configs["splatfacto"])
    config.output_dir = args.output.resolve()
    config.experiment_name = args.scene.name
    config.max_num_iterations = args.iterations
    config.pipeline.datamanager.data = args.scene.resolve()
    config.pipeline.datamanager.dataparser.downscale_factor = 1
    config.pipeline.datamanager.dataparser.eval_mode = "fraction"
    config.pipeline.datamanager.dataparser.train_split_fraction = 0.9
    config.pipeline.datamanager.cache_images = "cpu"
    config.pipeline.model.background_color = "white"
    config.pipeline.model.num_downscales = 1
    config.pipeline.model.resolution_schedule = 1000
    config.pipeline.model.camera_optimizer.mode = "off"
    from food3d.io import load_config
    quality = load_config(args.config).get('reconstruction', {})
    if not 0 <= float(quality.get('cull_alpha_thresh',0.1)) < 1:
        raise ValueError('cull_alpha_thresh must be in [0,1)')
    if float(quality.get('densify_grad_thresh',0.0008)) <= 0 or int(quality.get('stop_split_at',15000)) < 1:
        raise ValueError('Densification threshold and stop step must be positive')
    allowed = {'cull_alpha_thresh', 'densify_grad_thresh', 'stop_split_at', 'use_absgrad'}
    for key in allowed & quality.keys():
        if not hasattr(config.pipeline.model,key):
            raise RuntimeError(f'Installed Splatfacto lacks model parameter {key}')
        setattr(config.pipeline.model,key,quality[key])
    config.vis = "tensorboard"
    config.steps_per_save = min(args.iterations, 1000)
    config.steps_per_eval_image = 500
    config.steps_per_eval_all_images = 0
    config.logging.local_writer.enable = True
    config.machine.num_devices = 1
    config.machine.device_type = "cuda"
    config.set_timestamp()
    config.save_config()
    from food3d.io import read_json, write_json
    write_json(config.get_base_dir() / "input_provenance.json", {"transforms": read_json(args.scene / "transforms.json"), "legacy_colmap_layout_detected": (args.scene / "colmap/sparse/0").exists(), "source_scene": str(args.scene.resolve()), "quality_options": {k:quality[k] for k in allowed & quality.keys()}})
    train_loop(local_rank=0, world_size=1, config=config)


if __name__ == "__main__":
    main()
