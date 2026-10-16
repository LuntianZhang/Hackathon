"""Analytical ellipsoids and a table: geometry fixtures, not food recognition data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .data.cameras import export_nerfstudio_transforms
from .io import initialize_scene, write_json, write_point_cloud_ply
from .schemas import CameraFrame, save_scene

OBJECTS = [
    {"instance_id": 1, "name": "苹果", "center": [-0.25, -0.13, 0.095], "radii": [0.09, 0.085, 0.09], "color": [205, 45, 35]},
    {"instance_id": 2, "name": "香蕉", "center": [0.0, 0.18, 0.055], "radii": [0.18, 0.035, 0.045], "color": [235, 200, 40]},
    {"instance_id": 3, "name": "面包", "center": [0.27, -0.10, 0.07], "radii": [0.13, 0.08, 0.065], "color": [180, 115, 55]},
]


def look_at(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0., 0., 1.]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, down, forward])
    pose[:3, 3] = position
    return pose


def _render(camera: CameraFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.mgrid[:camera.height, :camera.width]
    local_rays = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3) @ np.linalg.inv(np.asarray(camera.K)).T
    pose = np.asarray(camera.camera_to_world)
    directions = local_rays @ pose[:3, :3].T
    origin = pose[:3, 3]
    best = np.full(len(directions), np.inf)
    labels = np.zeros(len(directions), dtype=np.uint16)
    rgb = np.full((len(directions), 3), [225, 230, 235], dtype=float)
    # Since local ray z=1, intersection parameter equals camera-z depth.
    table_depth = np.full(len(directions), np.inf)
    toward_table = directions[:, 2] < -1e-10
    table_depth[toward_table] = -origin[2] / directions[toward_table, 2]
    table_positions = origin + directions[toward_table] * table_depth[toward_table, None]
    table_valid = np.zeros(len(directions), dtype=bool)
    table_valid[toward_table] = (np.abs(table_positions[:, :2]) <= 0.8).all(axis=1)
    table_valid &= table_depth > 0
    best[table_valid] = table_depth[table_valid]
    positions = origin + directions[table_valid] * table_depth[table_valid, None]
    pattern = ((np.floor(positions[:, 0] * 12) + np.floor(positions[:, 1] * 12)) % 2)[:, None]
    rgb[table_valid] = np.array([155, 140, 120]) + pattern * 12
    for obj in OBJECTS:
        radii = np.asarray(obj["radii"])
        offset = (origin - np.asarray(obj["center"])) / radii
        scaled = directions / radii
        a = np.sum(scaled * scaled, axis=1)
        b = 2 * (scaled @ offset)
        c = float(offset @ offset) - 1
        discriminant = b*b - 4*a*c
        intersection = np.full(len(directions), np.inf)
        hits = discriminant >= 0
        near = (-b[hits] - np.sqrt(discriminant[hits])) / (2*a[hits])
        far = (-b[hits] + np.sqrt(discriminant[hits])) / (2*a[hits])
        intersection[hits] = np.where(near > 1e-8, near, np.where(far > 1e-8, far, np.inf))
        visible = intersection < best
        best[visible] = intersection[visible]
        labels[visible] = obj["instance_id"]
        hit_positions = origin + directions[visible] * intersection[visible, None]
        normal = (hit_positions - np.asarray(obj["center"])) / radii**2
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
        light = np.array([-0.4, -0.3, 0.8])
        light /= np.linalg.norm(light)
        shade = 0.45 + 0.55 * np.maximum(normal @ light, 0)
        rgb[visible] = np.asarray(obj["color"])[None, :] * shade[:, None]
    best[~np.isfinite(best)] = 0
    return np.clip(rgb, 0, 255).astype(np.uint8).reshape(camera.height, camera.width, 3), labels.reshape(camera.height, camera.width), best.astype(np.float32).reshape(camera.height, camera.width)


def generate_scene(scene: Path, options: dict) -> dict:
    scene = scene.resolve()
    views = int(options.get("views", 12))
    width, height = int(options.get("width", 320)), int(options.get("height", 240))
    seed = int(options.get("seed", 7))
    if views < 2 or min(width, height) < 32:
        raise ValueError("模拟场景至少需要 2 个视角，图像边长至少 32")
    initialize_scene(scene)
    for directory in ["images", "masks", "depth"]:
        (scene / directory).mkdir()
    focal = max(width, height) * 0.9
    k = [[focal, 0, (width-1)/2], [0, focal, (height-1)/2], [0, 0, 1]]
    frames = []
    for index, angle in enumerate(np.linspace(0, 2*np.pi, views, endpoint=False)):
        frame_id = f"{index:06d}"
        position = np.array([0.95*np.cos(angle), 0.95*np.sin(angle), 0.7])
        camera = CameraFrame(frame_id, f"images/{frame_id}.png", width, height, k, look_at(position, np.array([0, 0, 0.05])).tolist(), mask_path=f"masks/{frame_id}.png", depth_path=f"depth/{frame_id}.npy")
        camera.validate()
        rgb, mask, depth = _render(camera)
        Image.fromarray(rgb).save(scene / camera.image_path)
        Image.fromarray(mask).save(scene / camera.mask_path)
        np.save(scene / camera.depth_path, depth)
        frames.append(camera)
    random = np.random.default_rng(seed)
    point_sets, color_sets, label_sets = [], [], []
    for obj in OBJECTS:
        direction = random.normal(size=(600, 3))
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        points = np.asarray(obj["center"]) + direction * np.asarray(obj["radii"])
        point_sets.append(points)
        color_sets.append(np.tile(obj["color"], (len(points), 1)))
        label_sets.append(np.full(len(points), obj["instance_id"], dtype=np.uint16))
    tabletop = np.column_stack([random.uniform(-0.65, 0.65, (1000, 2)), np.zeros(1000)])
    point_sets.append(tabletop)
    color_sets.append(np.tile([155, 140, 120], (len(tabletop), 1)))
    label_sets.append(np.zeros(len(tabletop), dtype=np.uint16))
    points, colors, labels = np.concatenate(point_sets), np.concatenate(color_sets).astype(np.uint8), np.concatenate(label_sets)
    np.savez_compressed(scene / "sparse_points.npz", positions=points, colors=colors)
    write_point_cloud_ply(scene / "sparse_points.ply", points, colors)
    np.savez_compressed(scene / "ground_truth_points.npz", positions=points, instance_ids=labels)
    write_json(scene / "instances.json", {"schema_version": 1, "source": "synthetic_ground_truth", "instances": [{"instance_id": 0, "name": "背景"}, *[{"instance_id": obj["instance_id"], "name": obj["name"]} for obj in OBJECTS]], "unknown_id": 65535})
    write_json(scene / "synthetic_geometry.json", {"objects": OBJECTS, "table_z": 0, "seed": seed, "notice": "食物名称只是几何真值标签；形状是椭球，不能用于证明真实识别或重建效果。"})
    save_scene(scene / "scene.json", frames, source="synthetic", world_units="meters", metadata={"synthetic_geometry": "synthetic_geometry.json", "training_world_transform": "not_yet_applied", "depth_invalid_value": 0})
    export_nerfstudio_transforms(scene / "transforms.json", frames, "sparse_points.ply")
    result = {"source": "synthetic", "stage": "generated", "views": views, "points": len(points), "reconstruction_verified": False, "semantics_verified": False, "execution_status": "generated_requires_review", "notice": "解析几何数据，不是 COLMAP 或 3DGS 的输出。"}
    write_json(scene / "validation.json", result)
    return result

