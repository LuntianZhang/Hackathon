"""Only implemented stages are exposed as runnable commands."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .io import load_config, write_json
from .paths import resolve_arguments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="food3d", description="Food3D POC 数据、重建、语义融合与演示工具")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="读取 Python、GPU 和工具环境")
    doctor.add_argument("--output", type=Path)
    synthetic = commands.add_parser("synthetic", help="生成解析几何模拟数据")
    synthetic.add_argument("--scene", required=True, type=Path)
    synthetic.add_argument("--config", type=Path)
    synthetic.add_argument("--views", type=int)
    extract = commands.add_parser("extract", help="视频采样与帧质量筛选")
    extract.add_argument("--video", required=True, type=Path)
    extract.add_argument("--scene", required=True, type=Path)
    extract.add_argument("--config", type=Path)
    photos = commands.add_parser("photos", help="按文件名顺序导入照片目录")
    photos.add_argument("--input", required=True, type=Path)
    photos.add_argument("--scene", required=True, type=Path)
    photos.add_argument("--config", type=Path)
    photos.add_argument("--source-kind", choices=["photos", "rendered"], default="photos")
    colmap = commands.add_parser("colmap", help="COLMAP 位姿估计、去畸变、数据导出")
    colmap.add_argument("--scene", required=True, type=Path)
    colmap.add_argument("--config", type=Path)
    inspect = commands.add_parser("inspect", help="生成缩略图和点云投影诊断")
    inspect.add_argument("--scene", required=True, type=Path)
    inspect.add_argument("--max-points", type=int, default=1200)
    background=commands.add_parser('board-background',help='在相机匹配前跟踪定位板和完整物体，排除背景')
    background.add_argument('--scene',required=True,type=Path)
    background.add_argument('--checkpoint',type=Path,default=Path('.tools/weights/sam2.1_hiera_small.pt'))
    background.add_argument('--minimum-frames',type=int,default=2)
    train = commands.add_parser("train", help="在独立环境调用官方 Splatfacto")
    train.add_argument("--scene", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--python", type=Path, default=Path(".venv-gs/Scripts/python.exe"))
    train.add_argument("--iterations", type=int, default=3000)
    train.add_argument("--config", type=Path, help="Splatfacto quality parameter overrides")
    export = commands.add_parser("export", help="导出固定高斯参数和训练坐标相机")
    export.add_argument("--checkpoint", required=True, type=Path)
    export.add_argument("--scene", required=True, type=Path)
    export.add_argument("--output", required=True, type=Path)
    export.add_argument("--transform", required=True, type=Path)
    export.add_argument("--sh-degree", type=int, default=3)
    segment = commands.add_parser("segment", help="SAM 2 分割传播，支持清单、文字、人工与大模型多视图识别")
    segment.add_argument("--scene", required=True, type=Path)
    segment.add_argument("--output", required=True, type=Path)
    segment.add_argument("--checkpoint", type=Path, default=Path(".tools/weights/sam2.1_hiera_small.pt"))
    segment.add_argument("--reference-frame")
    segment.add_argument("--prompts", type=Path)
    segment.add_argument("--assistance", type=Path, help='清单、文字定位、点击框选或 mode=llm 大模型识别 JSON')
    segment.add_argument('--seed-masks', type=Path, help='传播选定结果的实际掩码，保留其名称和 ID')
    fuse = commands.add_parser("fuse", help="按可见高斯贡献融合多视角实例")
    fuse.add_argument("--bundle", required=True, type=Path)
    fuse.add_argument("--segmentation", required=True, type=Path)
    fuse.add_argument("--max-views", type=int, default=40)
    fuse.add_argument("--width", type=int, default=480)
    fuse.add_argument('--depth-reference',choices=['surface','expected'],default='surface')
    serve = commands.add_parser("serve", help="打开三维食物点击演示")
    serve.add_argument("--bundle", required=True, type=Path)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=7860)
    evaluate = commands.add_parser("evaluate", help="保留视角的三维语义一致性与点击评估")
    evaluate.add_argument("--bundle", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--width", type=int, default=640)
    volume = commands.add_parser("volume", help="按三维实例估算凸包包围体积，并可用实测尺寸标定")
    volume.add_argument("--bundle", required=True, type=Path)
    volume.add_argument("--reference-instance", type=int)
    volume.add_argument("--reference-span-cm", type=float)
    board=commands.add_parser('board-scale',help='用 PanSense 定位板自动标定真实体积尺度')
    board.add_argument('--bundle',required=True,type=Path)
    paper=commands.add_parser('paper-scale',help='用多视角 10×2 cm 白色纸片自动标定模型尺度')
    paper.add_argument('--bundle',required=True,type=Path)
    paper.add_argument('--length-cm',type=float,default=10)
    paper.add_argument('--width-cm',type=float,default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = resolve_arguments(build_parser().parse_args(argv))
    try:
        if args.command == "doctor":
            from .doctor import report
            result = report()
            if args.output:
                write_json(args.output, result)
        elif args.command == "synthetic":
            from .synthetic import generate_scene
            options = load_config(args.config).get("synthetic", {}).copy()
            if args.views is not None:
                options["views"] = args.views
            result = generate_scene(args.scene, options)
        elif args.command == "extract":
            from .data.frames import extract_video
            result = extract_video(args.video, args.scene, load_config(args.config).get("frames", {}))
            result = {"scene": str(args.scene.resolve()), "sampled": len(result["frames"]), "accepted": sum(frame["accepted"] for frame in result["frames"])}
        elif args.command == "colmap":
            from .data.colmap import reconstruct_cameras
            result = reconstruct_cameras(args.scene, load_config(args.config).get("colmap", {}))
        elif args.command == "photos":
            from .data.photos import import_photos
            result = import_photos(args.input, args.scene, load_config(args.config).get("photos", {}), args.source_kind)
        elif args.command == 'board-background':
            from .data.board_background import prepare_background
            result=prepare_background(args.scene,args.checkpoint,args.minimum_frames)
        elif args.command == "train":
            from .reconstruction.splatfacto import train
            result = train(args.scene, args.output, args.python, args.iterations, args.config)
        elif args.command == "export":
            from .reconstruction.export import export_checkpoint
            result = export_checkpoint(args.checkpoint, args.scene, args.output, args.transform, args.sh_degree)
        elif args.command == "segment":
            from .segmentation.sam2 import segment
            from .io import read_json
            result = segment(args.scene, args.output, args.checkpoint, args.reference_frame, args.prompts,
                             assistance=read_json(args.assistance) if args.assistance else None, seed_masks=args.seed_masks)
        elif args.command == "fuse":
            from .fusion.pipeline import fuse
            result = fuse(args.bundle, args.segmentation, args.max_views, args.width, depth_reference=args.depth_reference)
        elif args.command == "serve":
            from .viewer.app import serve
            serve(args.bundle, args.host, args.port)
            return 0
        elif args.command == "evaluate":
            from .evaluation import evaluate
            result = evaluate(args.bundle, args.output, args.width)
        elif args.command == 'board-scale':
            from .board_scale import calibrate_board
            result=calibrate_board(args.bundle)
        elif args.command == 'paper-scale':
            from .paper_scale import calibrate_paper
            result=calibrate_paper(args.bundle,args.length_cm,args.width_cm)
        elif args.command == "volume":
            from .volume import calibrate, estimate_bundle, load_calibration, scaled_volume
            if (args.reference_instance is None) != (args.reference_span_cm is None):
                raise ValueError("尺寸标定必须同时提供 --reference-instance 与 --reference-span-cm")
            result = estimate_bundle(args.bundle)
            calibration = (calibrate(args.bundle, result, args.reference_instance, args.reference_span_cm)
                           if args.reference_instance is not None else load_calibration(args.bundle, result))
            factor = None if calibration is None else calibration["cm_per_model_unit"]
            result = {**result, "calibration": calibration,
                      "instances": [{**entry, "volume_ml": scaled_volume(entry, factor)} for entry in result["instances"]]}
        else:
            from .inspect import inspect_scene
            result = inspect_scene(args.scene, args.max_points)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 1
