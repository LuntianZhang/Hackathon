from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..io import read_json
from ..reconstruction.gaussians import load_bundle
from ..schemas import CameraFrame, UNKNOWN_ID


def resize_camera(camera: CameraFrame, width: int) -> CameraFrame:
    height = max(1, round(camera.height * width / camera.width))
    sx, sy = width / camera.width, height / camera.height
    k = np.asarray(camera.K).copy()
    k[0, 0] *= sx
    k[1, 1] *= sy
    k[0, 2] = (k[0, 2] + 0.5)*sx - 0.5
    k[1, 2] = (k[1, 2] + 0.5)*sy - 0.5
    return replace(camera, width=width, height=height, K=k.tolist())


class GaussianRenderer:
    def __init__(self, bundle: Path):
        import torch
        from gsplat import rasterization
        if not torch.cuda.is_available():
            raise RuntimeError("高斯渲染需要可用的 CUDA GPU")
        self.torch, self.rasterization = torch, rasterization
        self.bundle = bundle.resolve()
        self.model, self.metadata = load_bundle(self.bundle)
        self.parameters = {key: torch.tensor(getattr(self.model, key), device="cuda", dtype=torch.float32) for key in ["means", "quats", "scales", "opacities", "sh_coeffs"]}
        self.semantic_ids = None
        self.semantic_confidence = None
        self.instance_values = None
        self.features = None
        semantics_path = self.bundle / "semantics" / "semantics.npz"
        if semantics_path.exists():
            metadata = read_json(self.bundle / "semantics" / "fusion.json")
            if metadata["model_sha256"] != self.metadata["model_sha256"]:
                raise ValueError("语义结果绑定到不同高斯模型")
            with np.load(semantics_path, allow_pickle=False) as data:
                if not np.array_equal(data["gaussian_ids"], self.model.gaussian_ids):
                    raise ValueError("高斯索引与语义结果不一致")
                self.semantic_ids = data["instance_ids"]
                self.semantic_confidence = data["confidence"].astype(np.float32)
            self.instance_values = np.unique(np.concatenate([self.semantic_ids, [0, UNKNOWN_ID]])).astype(np.uint16)
            n, k = len(self.model.means), len(self.instance_values)
            features = np.zeros((n, k+1), dtype=np.float32)
            channels = np.searchsorted(self.instance_values, self.semantic_ids)
            features[np.arange(n), channels] = 1
            features[:, -1] = self.semantic_confidence
            self.features = torch.tensor(features, device="cuda")

    def camera_tensors(self, camera: CameraFrame):
        camera.validate()
        view = np.linalg.inv(np.asarray(camera.camera_to_world))
        k = np.asarray(camera.K).copy()
        # gsplat samples at x+0.5,y+0.5, canonical cameras sample at integers.
        k[0, 2] += 0.5
        k[1, 2] += 0.5
        return self.torch.tensor(view[None], device="cuda", dtype=self.torch.float32), self.torch.tensor(k[None], device="cuda", dtype=self.torch.float32)

    def render_geometry(self, camera: CameraFrame, packed: bool = False):
        torch = self.torch
        view, k = self.camera_tensors(camera)
        with torch.inference_mode():
            rendered, alpha, meta = self.rasterization(means=self.parameters["means"], quats=self.parameters["quats"], scales=self.parameters["scales"], opacities=self.parameters["opacities"], colors=self.parameters["sh_coeffs"], viewmats=view, Ks=k, width=camera.width, height=camera.height, packed=packed, sh_degree=self.model.sh_degree, backgrounds=torch.ones((1, 3), device="cuda"), render_mode="RGB+ED")
        return rendered, alpha, meta

    def render(self, camera: CameraFrame, selected_id: int = 0) -> dict:
        torch = self.torch
        rendered, alpha, _ = self.render_geometry(camera)
        rgb = rendered[0, :, :, :3].clamp(0, 1)
        labels = np.full((camera.height, camera.width), UNKNOWN_ID, dtype=np.uint16)
        confidence = np.zeros((camera.height, camera.width), dtype=np.float32)
        if self.features is not None:
            view, k = self.camera_tensors(camera)
            background = torch.zeros((1, self.features.shape[1]), device="cuda")
            background[0, int(np.searchsorted(self.instance_values, 0))] = 1
            with torch.inference_mode():
                semantic, semantic_alpha, _ = self.rasterization(means=self.parameters["means"], quats=self.parameters["quats"], scales=self.parameters["scales"], opacities=self.parameters["opacities"], colors=self.features, viewmats=view, Ks=k, width=camera.width, height=camera.height, packed=True, backgrounds=background, render_mode="RGB")
            channels = semantic[0, :, :, :-1]
            top_two = torch.topk(channels, k=2, dim=-1)
            labels = self.instance_values[top_two.indices[:, :, 0].cpu().numpy()]
            gaussian_confidence = semantic[0, :, :, -1] / semantic_alpha[0, :, :, 0].clamp(min=1e-6)
            confidence = torch.minimum(top_two.values[:, :, 0], gaussian_confidence).clamp(0, 1).cpu().numpy()
            # Explicit background contribution includes the uncovered ray.
            is_background = labels == 0
            confidence[is_background] = top_two.values[:, :, 0].cpu().numpy()[is_background]
            margin = (top_two.values[:, :, 0] - top_two.values[:, :, 1]).cpu().numpy()
            labels[(confidence < 0.55) | (margin < 0.15)] = UNKNOWN_ID
            if selected_id > 0 and selected_id != UNKNOWN_ID and selected_id in self.instance_values:
                channel = int(np.searchsorted(self.instance_values, selected_id))
                influence = channels[:, :, channel].clamp(0, 1)[:, :, None]
                highlight = torch.tensor([0.15, 0.9, 1.0], device="cuda")
                rgb = rgb*(1-0.5*influence) + highlight*0.5*influence
        return {"rgb": (rgb*255).to(torch.uint8).cpu().numpy(), "instance_ids": labels, "confidence": confidence, "depth": rendered[0, :, :, 3].cpu().numpy(), "alpha": alpha[0, :, :, 0].cpu().numpy()}
