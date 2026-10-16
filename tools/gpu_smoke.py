"""Checks the installed gsplat kernels and alpha-contribution parity."""

import torch
import numpy as np
from gsplat import rasterization

from food3d.fusion.visibility import splat_contributions
from food3d.io import write_json
from food3d.schemas import CameraFrame
from pathlib import Path
from food3d.paths import ProjectPaths


def main():
    camera = CameraFrame("test", "", 64, 64, [[100, 0, 31.5], [0, 100, 31.5], [0, 0, 1]], np.eye(4).tolist())
    means = torch.tensor([[0, 0, 2], [0, 0, 3]], device="cuda", dtype=torch.float32)
    quats = torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]], device="cuda", dtype=torch.float32)
    scales = torch.full((2, 3), 0.12, device="cuda")
    opacities = torch.tensor([0.95, 0.95], device="cuda")
    colors = torch.tensor([[1, 0, 0], [0, 0, 1]], device="cuda", dtype=torch.float32)
    k = torch.tensor([[[100, 0, 32], [0, 100, 32], [0, 0, 1]]], device="cuda", dtype=torch.float32)
    view = torch.eye(4, device="cuda")[None]
    rgb, alpha, meta = rasterization(means, quats, scales, opacities, colors, view, k, 64, 64, packed=False, render_mode="RGB+ED")
    gaussian_ids, pixel_ids, weights = splat_contributions(None, camera, meta)
    reconstructed = torch.zeros(64*64, device="cuda")
    reconstructed.scatter_add_(0, pixel_ids, weights)
    error = float((reconstructed-alpha.reshape(-1)).abs().max())
    assert error < 5e-4, f"CUDA alpha compositing mismatch: {error}"
    center = (pixel_ids == 32*64+32)
    front = float(weights[center & (gaussian_ids == 0)].sum())
    back = float(weights[center & (gaussian_ids == 1)].sum())
    assert front > back*5, "Occluded Gaussian has excessive contribution"
    result = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "alpha_parity_max_error": error, "front_contribution": front, "back_contribution": back, "status": "passed"}
    write_json(ProjectPaths().resolve("data/outputs/gpu_smoke.json"), result)
    print(result)


if __name__ == "__main__":
    main()
