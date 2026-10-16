from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from ..io import initialize_scene, read_json, write_json
from ..schemas import load_scene, save_scene
from .gaussians import GaussianSet, file_hash


def export_gaussian_ply(path: Path, model: GaussianSet) -> None:
    """Standard 3DGS binary PLY: SH channels, log scales, opacity logits, wxyz."""
    n = len(model.means)
    rest = model.sh_coeffs[:, 1:, :].transpose(0, 2, 1).reshape(n, -1)
    names = ['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)]
    names += [f'f_rest_{i}' for i in range(rest.shape[1])] + ['opacity']
    names += [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)]
    opacity = np.clip(model.opacities, 1e-7, 1-1e-7)
    values = np.column_stack([model.means, np.zeros((n,3)), model.sh_coeffs[:,0,:], rest,
                              np.log(opacity/(1-opacity)), np.log(model.scales), model.quats]).astype('<f4')
    header = ['ply', 'format binary_little_endian 1.0', 'comment food3d standard 3DGS export', f'element vertex {n}']
    header += ['property float '+name for name in names] + ['end_header']
    with path.open('wb') as stream:
        stream.write(('\n'.join(header)+'\n').encode('ascii'))
        stream.write(values.tobytes())


def transform_camera(pose: np.ndarray, transform: np.ndarray, scale: float) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    if transform.shape == (3, 4):
        transform = np.vstack([transform, [0, 0, 0, 1]])
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("训练坐标变换无效")
    result = transform @ pose
    result[:3, 3] *= scale
    return result


def camera_world_transform(reported: dict, provenance: dict) -> tuple[np.ndarray, float]:
    """Remove already-applied/legacy transforms from Nerfstudio's original-space map."""
    metadata = provenance["transforms"]
    applied = metadata.get("applied_transform")
    if applied is None and provenance.get("legacy_colmap_layout_detected"):
        applied = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0]]
    reported_matrix = np.eye(4)
    reported_matrix[:3] = np.asarray(reported["transform"])
    applied_matrix = np.eye(4)
    if applied is not None:
        applied_matrix[:3] = np.asarray(applied)
    return reported_matrix @ np.linalg.inv(applied_matrix), float(reported["scale"])/float(metadata.get("applied_scale", 1))


def export_checkpoint(checkpoint: Path, source_scene: Path, output: Path, dataparser_transform: Path, sh_degree: int = 3) -> dict:
    import torch
    checkpoint, source_scene, output = checkpoint.resolve(), source_scene.resolve(), output.resolve()
    source_metadata, frames = load_scene(source_scene / "scene.json")
    transform_data = read_json(dataparser_transform)
    provenance_path = checkpoint.parent.parent / "input_provenance.json"
    provenance = read_json(provenance_path) if provenance_path.is_file() else {"transforms": read_json(source_scene / "transforms.json"), "legacy_colmap_layout_detected": (source_scene / "colmap/sparse/0").exists()}
    transform, scale = camera_world_transform(transform_data, provenance)
    effective_transform = {"transform": transform[:3].tolist(), "scale": scale}
    if not (checkpoint.parent.parent / "config.yml").is_file():
        raise ValueError("只接受带有 config.yml 的本地 Nerfstudio 训练目录检查点")
    # Nerfstudio optimizer state includes NumPy scalars; this adapter is for
    # trusted, locally generated training checkpoints, not downloaded pickle files.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    pipeline = state["pipeline"]
    def parameter(name: str) -> np.ndarray:
        keys = [key for key in pipeline if key.endswith("gauss_params." + name)]
        if len(keys) != 1:
            raise ValueError(f"检查点无法唯一定位高斯参数 {name}")
        return pipeline[keys[0]].detach().float().cpu().numpy()
    means, quats, logits = parameter("means"), parameter("quats"), parameter("opacities").reshape(-1)
    quats /= np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)
    if sh_degree == 0:
        dc = (1 / (1 + np.exp(-np.clip(parameter("features_dc"), -80, 80))) - 0.5) / 0.28209479177387814
        sh = dc[:, None, :]
    else:
        sh = np.concatenate([parameter("features_dc")[:, None, :], parameter("features_rest")], axis=1)
    model = GaussianSet(means, quats, np.exp(parameter("scales")), 1 / (1 + np.exp(-np.clip(logits, -80, 80))), sh, np.arange(len(means), dtype=np.int64))
    model.validate()
    initialize_scene(output)
    model.save(output / "gaussians.npz")
    export_gaussian_ply(output / "gaussians.ply", model)
    converted = []
    for frame in frames:
        image_path = Path(os.path.relpath(source_scene / frame.image_path, output)).as_posix()
        # Segmentation uses source_scene camera pixels; training only changes world coordinates.
        pose = transform_camera(np.asarray(frame.camera_to_world), transform, scale)
        converted.append(replace(frame, image_path=image_path, camera_to_world=pose.tolist(), mask_path=None, depth_path=None))
    save_scene(output / "scene.json", converted, source="splatfacto_export", world_units="normalized_training_world", metadata={"source_scene": str(source_scene), "input_source": source_metadata.get("metadata", {}).get("input_source", source_metadata["source"]), "training_world_transform": effective_transform, "reported_dataparser_transform": transform_data, "source_camera_ids_preserved": True})
    metadata = {"schema_version": 1, "source": "nerfstudio_splatfacto", "checkpoint": str(checkpoint), "checkpoint_sha256": file_hash(checkpoint), "model_sha256": file_hash(output / "gaussians.npz"), "gaussians": len(means), "step": int(state["step"]), "sh_degree": model.sh_degree, "quaternion_order": "wxyz", "scales": "physical_exp_log_scale", "opacities": "sigmoid_logits", "source_scene": str(source_scene), "training_transform": effective_transform, "reported_dataparser_transform": transform_data, "semantics_verified": False}
    write_json(output / "model.json", metadata)
    return metadata
