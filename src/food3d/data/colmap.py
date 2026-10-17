"""Invoke installed COLMAP, introspect CLI options, and import undistorted cameras."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from ..io import read_json, run_logged, write_json, write_point_cloud_ply
from ..schemas import CameraFrame, save_scene
from .cameras import export_nerfstudio_transforms, quaternion_to_rotation
from .board_background import feature_mask_path,apply_background


def _help(executable: str, command: str) -> str:
    result = subprocess.run([executable, command, "-h"], capture_output=True, text=True, errors="replace", timeout=30)
    return result.stdout + result.stderr


def _gpu_option(help_text: str, candidates: list[str]) -> str:
    for name in candidates:
        if name in help_text:
            return "--" + name
    raise RuntimeError("无法在 COLMAP 实际帮助中找到 GPU 开关；请检查该版本 API")


def read_cameras(path: Path) -> dict[int, tuple[int, int, np.ndarray]]:
    cameras = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        camera_id, model, width, height = int(fields[0]), fields[1], int(fields[2]), int(fields[3])
        params = list(map(float, fields[4:]))
        if model == "PINHOLE" and len(params) == 4:
            fx, fy, cx, cy = params
        elif model == "SIMPLE_PINHOLE" and len(params) == 3:
            fx, cx, cy = params
            fy = fx
        else:
            raise ValueError(f"预期去畸变后的 PINHOLE，相机 {camera_id} 实际为 {model}")
        # COLMAP's top-left pixel center is (0.5, 0.5); canonical centers are (0, 0).
        cameras[camera_id] = (width, height, np.array([[fx, 0, cx - 0.5], [0, fy, cy - 0.5], [0, 0, 1]]))
    return cameras


def read_images(path: Path) -> list[dict]:
    images = []
    lines = iter(path.read_text(encoding="utf-8").splitlines())
    for line in lines:
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split(maxsplit=9)
        if len(fields) != 10:
            raise ValueError(f"COLMAP images.txt 图像行无效：{line[:120]}")
        images.append({"image_id": int(fields[0]), "q": np.array(fields[1:5], dtype=float), "t": np.array(fields[5:8], dtype=float), "camera_id": int(fields[8]), "name": fields[9]})
        # A valid image has a second line of 2D observations, possibly empty.
        if next(lines, None) is None:
            raise ValueError("COLMAP 图像缺少二维观测行")
    return images


def read_points(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz, rgb, errors = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        xyz.append(list(map(float, fields[1:4])))
        rgb.append(list(map(int, fields[4:7])))
        errors.append(float(fields[7]))
    return np.asarray(xyz, dtype=float).reshape(-1, 3), np.asarray(rgb, dtype=np.uint8).reshape(-1, 3), np.asarray(errors, dtype=float)


def reconstruct_cameras(scene: Path, options: dict) -> dict:
    scene = scene.resolve()
    manifest = read_json(scene / "frame_manifest.json")
    accepted = [frame for frame in manifest["frames"] if frame["accepted"]]
    minimum = int(options.get("minimum_registered_frames", 8))
    if minimum < 2 or len(accepted) < minimum:
        raise ValueError(f"需要至少 {max(minimum, 2)} 张有效输入帧")
    executable = shutil.which(str(options.get("executable", "colmap")))
    backend = str(options.get("backend", "auto"))
    if backend not in {"auto", "cli", "pycolmap"}:
        raise ValueError("COLMAP backend 仅支持 auto、cli、pycolmap")
    if backend == "pycolmap" or (backend == "auto" and not executable):
        return _reconstruct_pycolmap(scene, options, manifest)
    if not executable:
        raise RuntimeError("未找到 COLMAP 可执行程序")
    work = scene / "colmap"
    if work.exists():
        raise FileExistsError("COLMAP 输出目录已存在；请创建新的场景目录后重新处理")
    if (scene / "scene.json").exists() or (scene / "images").exists():
        raise FileExistsError("场景已有相机或去畸变图像，拒绝覆盖")
    matcher = str(options.get("matcher", "sequential"))
    if matcher not in {"sequential", "exhaustive"}:
        raise ValueError("matcher 仅支持 sequential 或 exhaustive")
    gpu = "1" if options.get("use_gpu", False) else "0"
    feature_gpu = _gpu_option(_help(executable, "feature_extractor"), ["FeatureExtraction.use_gpu", "SiftExtraction.use_gpu"])
    match_command = matcher + "_matcher"
    match_gpu = _gpu_option(_help(executable, match_command), ["FeatureMatching.use_gpu", "SiftMatching.use_gpu"])
    work.mkdir()
    sparse = work / "sparse"
    sparse.mkdir()
    log = scene / "logs" / "colmap.log"
    database = work / "database.db"
    mask_root=feature_mask_path(scene,manifest)
    mask_args=['--ImageReader.mask_path',str(mask_root)] if mask_root else []
    run_logged([executable, "feature_extractor", "--database_path", str(database), "--image_path", str(scene / "frames"), "--ImageReader.camera_model", str(options.get("camera_model", "OPENCV")), "--ImageReader.single_camera", "1" if options.get("single_camera", True) else "0", feature_gpu, gpu,*mask_args], log)
    run_logged([executable, match_command, "--database_path", str(database), match_gpu, gpu], log)
    run_logged([executable, "mapper", "--database_path", str(database), "--image_path", str(scene / "frames"), "--output_path", str(sparse)], log)
    models = []
    for candidate in sorted(sparse.iterdir()):
        if candidate.is_dir() and (candidate / "images.bin").exists():
            text_path = work / "model_text" / candidate.name
            text_path.mkdir(parents=True)
            run_logged([executable, "model_converter", "--input_path", str(candidate), "--output_path", str(text_path), "--output_type", "TXT"], log)
            models.append((len(read_images(text_path / "images.txt")), candidate))
    if not models:
        raise RuntimeError(f"COLMAP 没有成功生成模型，检查 {log}")
    count, best = max(models, key=lambda entry: entry[0])
    write_json(scene / "colmap_selection.json", {"selected_model": best.relative_to(scene).as_posix(), "registered_frames": count, "models": [{"path": model.relative_to(scene).as_posix(), "registered_frames": size} for size, model in models]})
    if count < minimum:
        raise RuntimeError(f"最大模型仅注册 {count} 帧，小于要求 {minimum}；检查 colmap_selection.json 和日志")
    dense = work / "undistorted"
    run_logged([executable, "image_undistorter", "--image_path", str(scene / "frames"), "--input_path", str(best), "--output_path", str(dense), "--output_type", "COLMAP"], log)
    text_path = work / "undistorted_text"
    text_path.mkdir()
    run_logged([executable, "model_converter", "--input_path", str(dense / "sparse"), "--output_path", str(text_path), "--output_type", "TXT"], log)
    return _export_scene_data(scene, manifest, text_path, dense,work/'model_text'/best.name)


def _export_scene_data(scene: Path, manifest: dict, text_path: Path, dense: Path, original_model=None) -> dict:
    accepted = [frame for frame in manifest["frames"] if frame["accepted"]]
    shutil.copytree(dense / "images", scene / "images")
    cameras = read_cameras(text_path / "cameras.txt")
    images = read_images(text_path / "images.txt")
    by_name = {Path(frame["image_path"]).name: frame for frame in accepted}
    frames = []
    for image in sorted(images, key=lambda entry: entry["name"]):
        name = image["name"]
        if name not in by_name:
            raise ValueError(f"COLMAP 图像 {name} 无法关联到抽帧清单")
        original = by_name[name]
        width, height, k = cameras[image["camera_id"]]
        relative_image = Path("images") / name
        with Image.open(scene / relative_image) as picture:
            if picture.size != (width, height):
                raise ValueError(f"去畸变图像 {name} 与内参尺寸不一致")
        rotation = quaternion_to_rotation(image["q"])
        pose = np.eye(4)
        pose[:3, :3] = rotation.T
        pose[:3, 3] = -rotation.T @ image["t"]
        frames.append(CameraFrame(original["frame_id"], relative_image.as_posix(), width, height, k.tolist(), pose.tolist(), source_frame_id=original["frame_id"], timestamp_seconds=original["timestamp_seconds"], colmap_image_id=image["image_id"], colmap_camera_id=image["camera_id"]))
    frames=apply_background(scene,manifest,frames,original_model)
    points, colors, errors = read_points(text_path / "points3D.txt")
    np.savez_compressed(scene / "sparse_points.npz", positions=points, colors=colors, reprojection_errors=errors)
    write_point_cloud_ply(scene / "sparse_points.ply", points, colors)
    registered_names = {image["name"] for image in images}
    statistics = {"source": manifest.get("source_type", "real_video"), "stage": "colmap", "input_frames": len(accepted), "registered_frames": len(frames), "registration_ratio": len(frames) / len(accepted), "unregistered_frame_ids": [frame["frame_id"] for frame in accepted if Path(frame["image_path"]).name not in registered_names], "sparse_points": len(points), "mean_reprojection_error_pixels": float(errors.mean()) if len(errors) else None, "reconstruction_verified": False, "semantics_verified": False, "execution_status": "colmap_completed_requires_visual_review"}
    save_scene(scene / "scene.json", frames, source="colmap", world_units="arbitrary_sfm", metadata={"input_source": statistics["source"], "training_world_transform": "not_yet_applied", "pixel_conversion": "COLMAP principal point minus 0.5", "frame_manifest": "frame_manifest.json"})
    export_nerfstudio_transforms(scene / "transforms.json", frames, "sparse_points.ply" if len(points) else None)
    write_json(scene / "validation.json", statistics)
    return statistics


def _reconstruct_pycolmap(scene: Path, options: dict, manifest: dict) -> dict:
    try:
        import pycolmap as p
    except ImportError as error:
        raise RuntimeError("没有 COLMAP CLI；请安装项目的 sfm extra：pip install -e '.[sfm]' ") from error
    work = scene / "colmap"
    if work.exists() or (scene / "scene.json").exists() or (scene / "images").exists():
        raise FileExistsError("场景已有 COLMAP 结果，拒绝覆盖；请使用新的场景目录")
    work.mkdir()
    sparse = work / "sparse"
    sparse.mkdir()
    threads = int(options.get("num_threads", 8))
    database = work / "database.db"
    def progress(stage: str):
        write_json(work / "progress.json", {"backend": "pycolmap", "version": p.__version__, "stage": stage})
        print(f"COLMAP: {stage}", flush=True)
    reader = p.ImageReaderOptions()
    reader.camera_model = str(options.get("camera_model", "OPENCV"))
    mask_root=feature_mask_path(scene,manifest)
    if mask_root:reader.mask_path=mask_root
    # These names were checked against the installed pycolmap 4.2.1 API.
    if not hasattr(p, "FeatureExtractionOptions"):
        raise RuntimeError("当前 pycolmap 版本不支持 FeatureExtractionOptions，请升级至验证版本")
    extraction = p.FeatureExtractionOptions()
    extraction.num_threads = threads
    extraction.use_gpu = bool(options.get("use_gpu", False))
    device = p.Device.cuda if extraction.use_gpu else p.Device.cpu
    reuse_database = options.get("reuse_database")
    if reuse_database and mask_root:
        raise ValueError('排除背景需要重新提取特征，不能复用旧的匹配数据库')
    if reuse_database:
        previous_database = Path(reuse_database).resolve()
        if not previous_database.is_file():
            raise FileNotFoundError(f"复用数据库不存在：{previous_database}")
        shutil.copy2(previous_database, database)
        with p.Database.open(database) as db:
            names = {image.name for image in db.read_all_images()}
        expected = {Path(frame["image_path"]).name for frame in manifest["frames"] if frame["accepted"]}
        if names != expected:
            raise ValueError("复用数据库图像清单与当前有效帧不一致")
        progress("reused_feature_database")
    else:
        progress("feature_extraction")
        p.extract_features(database_path=database, image_path=scene / "frames", camera_mode=p.CameraMode.SINGLE if options.get("single_camera", True) else p.CameraMode.PER_IMAGE, reader_options=reader, extraction_options=extraction, device=device)
    matching = p.FeatureMatchingOptions()
    matching.num_threads = threads
    matching.use_gpu = extraction.use_gpu
    matching.guided_matching = bool(options.get("guided_matching", False))
    matcher = options.get("matcher", "sequential")
    progress("feature_matching")
    if options.get("reuse_verified_matches", False):
        if not reuse_database:
            raise ValueError("复用已验证匹配需要指定 reuse_database")
        progress("reused_verified_matches")
    elif matcher == "sequential":
        pairing = p.SequentialPairingOptions()
        pairing.overlap = int(options.get("sequential_overlap", 10))
        p.match_sequential(database, matching_options=matching, pairing_options=pairing, device=device)
    elif matcher == "exhaustive":
        p.match_exhaustive(database, matching_options=matching, device=device)
    else:
        raise ValueError("matcher 仅支持 sequential 或 exhaustive")
    pipeline = p.IncrementalPipelineOptions()
    pipeline.num_threads = threads
    pipeline.random_seed = 7
    pipeline.ba_refine_focal_length = bool(options.get("refine_focal_length", True))
    pipeline.mapper.abs_pose_refine_focal_length = pipeline.ba_refine_focal_length
    pipeline.structure_less_registration_fallback = bool(options.get("structure_less_registration_fallback", True))
    pipeline.mapper.init_min_tri_angle = float(options.get("init_min_tri_angle", 16))
    pipeline.min_model_size = int(options.get("minimum_registered_frames", 8))
    pipeline.max_runtime_seconds = int(options.get("max_runtime_seconds", 900))
    progress("incremental_mapping")
    models = p.incremental_mapping(database, scene / "frames", sparse, options=pipeline)
    if not models:
        raise RuntimeError("COLMAP 未得到可注册的模型；检查纹理、视角重叠及几何匹配")
    best_id, best = max(models.items(), key=lambda pair: pair[1].num_reg_images())
    best_path = sparse / str(best_id)
    best_path.mkdir(exist_ok=True)
    best.write(best_path)
    count = best.num_reg_images()
    write_json(scene / "colmap_selection.json", {"backend": "pycolmap", "version": p.__version__, "selected_model": best_path.relative_to(scene).as_posix(), "registered_frames": count, "models": [{"model_id": model_id, "registered_frames": model.num_reg_images(), "points": model.num_points3D()} for model_id, model in models.items()]})
    if count < int(options.get("minimum_registered_frames", 8)):
        raise RuntimeError(f"最大模型仅注册 {count} 帧，未达到要求")
    dense = work / "undistorted"
    progress("undistortion")
    p.undistort_images(output_path=dense, input_path=best_path, image_path=scene / "frames", output_type="COLMAP", num_threads=threads)
    text_path = work / "undistorted_text"
    text_path.mkdir()
    p.Reconstruction(dense / "sparse").write_text(text_path)
    statistics = _export_scene_data(scene, manifest, text_path, dense,best_path)
    progress("complete")
    return statistics

