from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..io import read_json


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class GaussianSet:
    means: np.ndarray
    quats: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    sh_coeffs: np.ndarray
    gaussian_ids: np.ndarray

    def validate(self) -> None:
        n = len(self.means)
        if not n or self.means.shape != (n, 3) or self.quats.shape != (n, 4) or self.scales.shape != (n, 3) or self.opacities.shape != (n,) or self.sh_coeffs.ndim != 3 or self.sh_coeffs.shape[0] != n or self.sh_coeffs.shape[2] != 3 or self.gaussian_ids.shape != (n,):
            raise ValueError("高斯参数形状无效")
        for value in [self.means, self.quats, self.scales, self.opacities, self.sh_coeffs]:
            if not np.isfinite(value).all():
                raise ValueError("高斯参数含非有限数值")
        if (self.scales <= 0).any() or (self.opacities < 0).any() or (self.opacities > 1).any():
            raise ValueError("高斯尺度或透明度无效")
        if not np.allclose(np.linalg.norm(self.quats, axis=1), 1, atol=1e-4):
            raise ValueError("高斯四元数必须归一化")
        if len(np.unique(self.gaussian_ids)) != n:
            raise ValueError("gaussian_ids 必须唯一")
        degree = int(round(np.sqrt(self.sh_coeffs.shape[1]))) - 1
        if (degree + 1)**2 != self.sh_coeffs.shape[1]:
            raise ValueError("SH 系数数量必须为平方数")

    @property
    def sh_degree(self) -> int:
        return int(round(np.sqrt(self.sh_coeffs.shape[1]))) - 1

    def save(self, path: Path) -> None:
        self.validate()
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path: Path) -> "GaussianSet":
        with np.load(path, allow_pickle=False) as data:
            model = cls(**{key: data[key] for key in ["means", "quats", "scales", "opacities", "sh_coeffs", "gaussian_ids"]})
        model.validate()
        return model


def load_bundle(path: Path) -> tuple[GaussianSet, dict]:
    metadata = read_json(path / "model.json")
    model_path = path / "gaussians.npz"
    if metadata.get("model_sha256") != file_hash(model_path):
        raise ValueError("高斯模型哈希与 model.json 不一致")
    return GaussianSet.load(model_path), metadata
