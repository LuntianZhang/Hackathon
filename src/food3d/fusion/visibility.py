from __future__ import annotations

import numpy as np


def front_to_back_weights(pixel_ids: np.ndarray, depths: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """CPU reference for grouped alpha compositing; output preserves input order."""
    pixel_ids, depths, alphas = np.asarray(pixel_ids), np.asarray(depths), np.asarray(alphas)
    if pixel_ids.shape != depths.shape or depths.shape != alphas.shape or alphas.ndim != 1:
        raise ValueError("贡献索引、深度、alpha 形状必须一致")
    if not np.isfinite(alphas).all() or not np.isfinite(depths).all() or (alphas < 0).any() or (alphas >= 1).any():
        raise ValueError("alpha 必须在 [0,1) 范围且输入必须有限")
    result = np.zeros_like(alphas, dtype=float)
    order = np.lexsort((depths, pixel_ids))
    last_pixel, transmittance = None, 1.0
    for index in order:
        if pixel_ids[index] != last_pixel:
            last_pixel, transmittance = pixel_ids[index], 1.0
        result[index] = transmittance*alphas[index]
        transmittance *= 1-alphas[index]
    return result


def splat_contributions(renderer, camera, meta):
    """Use gsplat's CUDA intersection query and reconstruct T*alpha per pixel."""
    import torch
    from gsplat.cuda._wrapper import rasterize_to_indices_in_range
    with torch.inference_mode():
        gaussian_ids, pixel_ids, camera_ids = rasterize_to_indices_in_range(0, 1000000, torch.ones((1, camera.height, camera.width), device="cuda"), meta["means2d"], meta["conics"], meta["opacities"], camera.width, camera.height, meta["tile_size"], meta["isect_offsets"], meta["flatten_ids"])
        if gaussian_ids.numel() == 0:
            return gaussian_ids.long(), pixel_ids.long(), torch.empty(0, device="cuda")
        gaussian_ids, pixel_ids = gaussian_ids.long(), pixel_ids.long()
        depth_order = torch.argsort(meta["depths"][0], stable=True)
        rank = torch.empty_like(depth_order)
        rank[depth_order] = torch.arange(len(rank), device="cuda")
        order = torch.argsort(pixel_ids*len(rank) + rank[gaussian_ids], stable=True)
        gaussian_ids, pixel_ids = gaussian_ids[order], pixel_ids[order]
        xy = torch.stack([(pixel_ids % camera.width).float()+0.5, (pixel_ids // camera.width).float()+0.5], dim=-1)
        delta = xy - meta["means2d"][0, gaussian_ids]
        conic = meta["conics"][0, gaussian_ids]
        exponent = 0.5*(conic[:, 0]*delta[:, 0]**2 + conic[:, 2]*delta[:, 1]**2) + conic[:, 1]*delta[:, 0]*delta[:, 1]
        alpha = (meta["opacities"][0, gaussian_ids] * torch.exp(-exponent)).clamp(0, 0.999)
        logs = torch.log1p(-alpha).double()
        prefix = torch.cumsum(logs, dim=0)-logs
        starts = torch.ones(len(pixel_ids), dtype=torch.bool, device="cuda")
        starts[1:] = pixel_ids[1:] != pixel_ids[:-1]
        groups = starts.long().cumsum(0)-1
        offsets = prefix[torch.nonzero(starts, as_tuple=True)[0]]
        weights = torch.exp(prefix-offsets[groups]).float()*alpha
        return gaussian_ids, pixel_ids, weights


def pixel_surface_depths(pixel_ids, depths, contributions, pixel_count: int):
    """First depth reaching 50% of rendered alpha mass, not mean layer depth.

    Inputs must be sorted by pixel and then front-to-back, as returned by
    splat_contributions. Background leakage must not shift the visible surface.
    """
    import torch
    with torch.inference_mode():
        total = torch.zeros(pixel_count, device=depths.device)
        total.scatter_add_(0,pixel_ids,contributions)
        if not len(pixel_ids):
            return total
        mass = contributions.double()
        prefix = torch.cumsum(mass,0)-mass
        starts = torch.ones(len(pixel_ids),dtype=torch.bool,device=depths.device)
        starts[1:] = pixel_ids[1:] != pixel_ids[:-1]
        groups = starts.long().cumsum(0)-1
        offsets = prefix[torch.nonzero(starts,as_tuple=True)[0]]
        accumulated = prefix-offsets[groups]+mass
        crossing = accumulated >= 0.5*total[pixel_ids]
        candidate = torch.where(crossing,depths,torch.inf)
        surface = torch.full((pixel_count,),torch.inf,device=depths.device)
        surface.scatter_reduce_(0,pixel_ids,candidate,reduce='amin',include_self=True)
        return torch.where(torch.isfinite(surface) & (total > 1e-6),surface,0)
