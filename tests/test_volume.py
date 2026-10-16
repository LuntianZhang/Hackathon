from itertools import product
from pathlib import Path

import numpy as np
import pytest

from food3d.io import write_json
from food3d.reconstruction.gaussians import GaussianSet, file_hash
from food3d.volume import (calibrate, estimate_bundle, estimate_points,
                           load_calibration, scaled_volume, volume_text)


def cube():
    return np.array(list(product(np.linspace(0, 1, 5), repeat=3)))


def test_cube_volume_translation_rotation_and_cubic_scale():
    points = cube()
    angle = .73
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    estimate = estimate_points(points @ rotation.T*3 + [100, -40, 200])
    assert estimate["volume_model_units3"] == pytest.approx(27)
    assert estimate["max_span_model_units"] == pytest.approx(3*np.sqrt(3))
    assert scaled_volume(estimate, 2) == pytest.approx(216)
    assert scaled_volume(estimate, None) is None


def test_sphere_surface_recovers_solid_volume_without_summing_kernels():
    count = 2000
    z = 1-2*(np.arange(count)+.5)/count
    theta = np.arange(count)*np.pi*(3-np.sqrt(5))
    points = np.column_stack([np.sqrt(1-z*z)*np.cos(theta), np.sqrt(1-z*z)*np.sin(theta), z])
    assert estimate_points(points)["volume_model_units3"] == pytest.approx(4*np.pi/3, rel=.01)
    # Duplicating surface splats must not double a solid object's volume.
    assert estimate_points(np.repeat(points, 2, axis=0))["volume_model_units3"] == pytest.approx(estimate_points(points)["volume_model_units3"])


def test_isolated_outlier_does_not_inflate_cube_and_planar_is_rejected():
    entry = estimate_points(np.vstack([cube(), [30, 40, 50]]))
    assert entry["removed_outliers"] == 1
    assert entry["volume_model_units3"] == pytest.approx(1)
    points = cube()
    points[:, 2] = 0
    assert estimate_points(points)["volume_model_units3"] is None
    assert estimate_points(points[:3])["status"] == "insufficient_geometry"


def make_bundle(path: Path):
    path.mkdir()
    points = cube()
    n = len(points)
    model = GaussianSet(points, np.tile([1., 0, 0, 0], (n, 1)), np.ones((n, 3))*.01,
                        np.ones(n)*.9, np.zeros((n, 1, 3)), np.arange(n))
    model.save(path/"gaussians.npz")
    digest = file_hash(path/"gaussians.npz")
    write_json(path/"model.json", {"model_sha256": digest})
    write_json(path/"semantics"/"fusion.json", {"model_sha256": digest})
    np.savez(path/"semantics"/"semantics.npz", gaussian_ids=model.gaussian_ids,
             instance_ids=np.ones(n, dtype=np.uint16), confidence=np.ones(n), support_views=np.ones(n)*4)


def test_bundle_cache_calibration_and_semantic_invalidation(tmp_path):
    bundle = tmp_path/"bundle"
    make_bundle(bundle)
    report = estimate_bundle(bundle)
    assert [e["instance_id"] for e in report["instances"]] == [1]
    assert report == estimate_bundle(bundle)
    assert "未标定" in volume_text(report, 1, None)
    calibration = calibrate(bundle, report, 1, 10*np.sqrt(3))
    assert calibration["cm_per_model_unit"] == pytest.approx(10)
    assert scaled_volume(report["instances"][0], 10) == pytest.approx(1000)
    assert "mL" in volume_text(report, 1, calibration)
    assert "请选择" in volume_text(report, 0, calibration)
    assert load_calibration(bundle, report) == calibration
    semantic_path = bundle/"semantics"/"semantics.npz"
    with np.load(semantic_path) as data:
        values = dict(data)
    values["instance_ids"][:] = 0
    np.savez(semantic_path, **values)
    new_report = estimate_bundle(bundle)
    assert new_report["instances"] == []
    assert load_calibration(bundle, new_report) is None
    # A model mismatch is rejected instead of using a cached measurement.
    write_json(bundle/"semantics"/"fusion.json", {"model_sha256": "different"})
    with pytest.raises(ValueError, match="不匹配"):
        estimate_bundle(bundle)


@pytest.mark.parametrize("value", [0, -2, float("nan"), float("inf")])
def test_invalid_metric_scale_rejected(value):
    with pytest.raises(ValueError):
        scaled_volume({"volume_model_units3": 1}, value)
