from __future__ import annotations

import numpy as np

from ..schemas import UNKNOWN_ID


def weight_view_evidence(raw_votes: np.ndarray, minimum_mass: float = 0.001) -> tuple[np.ndarray, np.ndarray]:
    """Count consistent visible views independently from individual splat opacity.

    Preserve T*alpha mass in vote weight. A low-opacity Gaussian can receive
    several independent supporting views without having a large pixel footprint.
    Negligible transmitted evidence never qualifies as a supporting view.
    """
    raw_votes = np.asarray(raw_votes, dtype=np.float32)
    if raw_votes.ndim != 2 or not np.isfinite(raw_votes).all() or (raw_votes < 0).any() or minimum_mass <= 0:
        raise ValueError('Invalid per-view visibility evidence')
    mass = raw_votes.sum(axis=1)
    distribution = raw_votes / np.maximum(mass[:,None],1e-12)
    weights = raw_votes / np.maximum(mass[:,None],0.02)
    support = (distribution >= 0.5) & (mass[:,None] >= minimum_mass)
    return weights,support.astype(np.uint16)


def finalize_votes(votes: np.ndarray, support: np.ndarray, instance_values: np.ndarray, minimum_views: int = 3, minimum_confidence: float = 0.65, minimum_margin: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if votes.shape != support.shape or votes.ndim != 2 or votes.shape[1] != len(instance_values) or votes.shape[1] < 2:
        raise ValueError("投票和支持视角矩阵形状无效")
    if not np.isfinite(votes).all() or (votes < 0).any():
        raise ValueError("投票权重必须有限且非负")
    total = votes.sum(axis=1)
    order = np.argsort(votes, axis=1)
    best = order[:, -1]
    row = np.arange(len(votes))
    confidence = votes[row, best] / np.maximum(total, 1e-12)
    margin = (votes[row, best] - votes[row, order[:, -2]]) / np.maximum(total, 1e-12)
    views = support[row, best]
    labels = instance_values[best].astype(np.uint16)
    unreliable = (total <= 0) | (views < minimum_views) | (confidence < minimum_confidence) | (margin < minimum_margin)
    labels[unreliable] = UNKNOWN_ID
    confidence[total <= 0] = 0
    return labels, confidence.astype(np.float32), views.astype(np.uint16)
