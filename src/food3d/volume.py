"""Instance convex-envelope estimates, in trained model coordinates.

Gaussian kernels overlap and describe appearance, not solid material. We use
their reliable centers as surface samples; never sum kernel ellipsoid volumes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree
from scipy.spatial.distance import cdist

from .io import read_json, write_json
from .reconstruction.gaussians import file_hash, load_bundle
from .schemas import UNKNOWN_ID

OPTIONS = {"confidence_min": 0.7, "opacity_min": 0.1, "support_min": 3,
           "outlier_mad_multiplier": 6.0, "minimum_points": 16}
METHOD = "reliable_center_convex_hull_v1"


def estimate_points(points: np.ndarray) -> dict:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("体积采样点必须是有限的 N×3 坐标")
    points = np.unique(points, axis=0)
    result = {"status": "insufficient_geometry", "unique_points": len(points),
              "retained_points": len(points), "removed_outliers": 0,
              "volume_model_units3": None, "max_span_model_units": None}
    if len(points) < OPTIONS["minimum_points"]:
        return result
    # Sixth-neighbor distance rejects isolated splats without clipping every
    # extremity by a coordinate percentile. Work in normalized coordinates.
    origin = np.median(points, axis=0)
    extent = float(np.max(np.ptp(points, axis=0)))
    if extent <= 0:
        return result
    normalized = (points-origin)/extent
    distance = cKDTree(normalized).query(normalized, k=7)[0][:, -1]
    median = float(np.median(distance))
    mad = float(np.median(np.abs(distance-median)))
    cutoff = median + OPTIONS["outlier_mad_multiplier"]*max(1.4826*mad, median*0.1)
    kept = points[distance <= cutoff]
    result.update(retained_points=len(kept), removed_outliers=len(points)-len(kept))
    if len(kept) < OPTIONS["minimum_points"]:
        return result
    # Recenter/rescale after outlier removal to avoid Qhull numerical problems.
    origin = np.mean(kept, axis=0)
    extent = float(np.max(np.ptp(kept, axis=0)))
    normalized = (kept-origin)/extent
    if np.linalg.matrix_rank(normalized, tol=1e-8) < 3:
        result["status"] = "degenerate_geometry"
        return result
    try:
        hull = ConvexHull(normalized)
    except QhullError:
        result["status"] = "degenerate_geometry"
        return result
    vertices = normalized[hull.vertices]
    # Exact diameter of the retained convex envelope, with bounded working RAM.
    diameter = max(float(cdist(vertices[start:start+256], vertices).max())
                   for start in range(0, len(vertices), 256))
    result.update(status="estimated", volume_model_units3=float(hull.volume*extent**3),
                  max_span_model_units=diameter*extent,
                  axis_extents_model_units=np.ptp(kept, axis=0).tolist(),
                  hull_vertices=len(vertices))
    return result


def estimate_bundle(bundle: Path, use_cache: bool = True) -> dict:
    bundle = bundle.resolve()
    model, metadata = load_bundle(bundle)
    semantic_path = bundle/"semantics"/"semantics.npz"
    fusion = read_json(bundle/"semantics"/"fusion.json")
    if fusion["model_sha256"] != metadata["model_sha256"]:
        raise ValueError("体积估算的语义与高斯模型不匹配")
    semantic_hash = file_hash(semantic_path)
    identity = {"model_sha256": metadata["model_sha256"], "semantics_sha256": semantic_hash,
                "method": METHOD, "options": OPTIONS}
    path = bundle/"measurements"/"volumes.json"
    if use_cache and path.exists():
        cached = read_json(path)
        if all(cached.get(key) == value for key, value in identity.items()):
            return cached
    with np.load(semantic_path, allow_pickle=False) as data:
        if not np.array_equal(data["gaussian_ids"], model.gaussian_ids):
            raise ValueError("体积估算的高斯索引与语义不匹配")
        ids, confidence, support = data["instance_ids"], data["confidence"], data["support_views"]
    entries = []
    for instance_id in np.unique(ids):
        if instance_id in (0, UNKNOWN_ID):
            continue
        mask = ((ids == instance_id) & (confidence >= OPTIONS["confidence_min"])
                & (support >= OPTIONS["support_min"]) & (model.opacities >= OPTIONS["opacity_min"]))
        entry = estimate_points(model.means[mask])
        entry.update(instance_id=int(instance_id), labeled_gaussians=int((ids == instance_id).sum()),
                     eligible_gaussians=int(mask.sum()))
        entries.append(entry)
    result = {"schema_version": 1, **identity, "unit": "model_unit_cubed",
              "instances": entries, "limitations": [
                  "Convex envelope includes concavities; curved bananas can be overestimated",
                  "Incomplete surfaces and semantic omissions can underestimate the envelope",
                  "Outlier filtering is heuristic and cannot certify volume accuracy",
                  "Metric scale requires a user-measured reference; no physical volume ground truth"]}
    write_json(path, result)
    return result


def scaled_volume(entry: dict, cm_per_unit: float | None) -> float | None:
    if cm_per_unit is not None and (not np.isfinite(cm_per_unit) or cm_per_unit <= 0):
        raise ValueError("每模型单位对应的厘米数必须为有限正数")
    value = entry["volume_model_units3"]
    return None if value is None or cm_per_unit is None else float(value*cm_per_unit**3)


def load_calibration(bundle: Path, report: dict) -> dict | None:
    path = bundle/"measurements"/"calibration.json"
    if not path.exists():
        return None
    value = read_json(path)
    if value.get('source')=='white_paper_rectangle_multiview':
        from .paper_scale import camera_hash
        if value.get('status')!='calibrated' or value.get('model_sha256')!=report['model_sha256'] or value.get('camera_sha256')!=camera_hash(bundle):
            return None
        scaled_volume({'volume_model_units3':1.0},value['cm_per_model_unit'])
        return value
    if any(value.get(key) != report[key] for key in ("model_sha256", "semantics_sha256", "method", "options")):
        return None
    scaled_volume({"volume_model_units3": 1.0}, value["cm_per_model_unit"])
    return value


def calibrate(bundle: Path, report: dict, instance_id: int, measured_span_cm: float) -> dict:
    if not np.isfinite(measured_span_cm) or measured_span_cm <= 0:
        raise ValueError("请输入实际测量的最大跨度，单位厘米，且必须大于零")
    entry = next((item for item in report["instances"] if item["instance_id"] == instance_id), None)
    if entry is None or entry["status"] != "estimated":
        raise ValueError("请先选中一个具有有效三维体积估算的食物")
    factor = float(measured_span_cm/entry["max_span_model_units"])
    scaled_volume(entry, factor)
    value = {"schema_version": 1,
             **{key: report[key] for key in ("model_sha256", "semantics_sha256", "method", "options")},
             "source": "user_measured_maximum_straight_line_span",
             "reference_instance_id": instance_id, "measured_span_cm": float(measured_span_cm),
             "reference_span_model_units": entry["max_span_model_units"], "cm_per_model_unit": factor}
    write_json(bundle/"measurements"/"calibration.json", value)
    return value


def volume_text(report: dict, selected: int, calibration: dict | None) -> str:
    entry = next((item for item in report["instances"] if item["instance_id"] == selected), None)
    if entry is None:
        return "请选择食物查看体积估算"
    if entry["status"] != "estimated":
        return "可靠三维采样不足，暂时无法估算体积"
    factor = None if calibration is None else calibration["cm_per_model_unit"]
    ml = scaled_volume(entry, factor)
    basis='基于白色纸片标定' if calibration and calibration.get('source')=='white_paper_rectangle_multiview' else '基于输入尺寸标定'
    value = (f"约 {ml:.1f} mL（cm³，{basis}）" if ml is not None
             else f"约 {entry['volume_model_units3']:.6g} 模型单位³（未标定）")
    span = entry["max_span_model_units"]
    span_text = f"{span*factor:.2f} cm" if factor is not None else f"{span:.5g} 模型单位"
    return (f"{value}\n凸包最大跨度：{span_text}\n"
            f"可靠采样：{entry['retained_points']} 个；剔除孤立点：{entry['removed_outliers']} 个\n"
            "包围体积粗估；香蕉弯曲空隙也会计入，缺失表面可能造成低估。")


def published_measurements(bundle: Path):
    """Read worker-produced measurements without rebuilding hulls in HTTP requests."""
    bundle=Path(bundle);path=bundle/'measurements/volumes.json'
    if not path.is_file(): return {'status':'unavailable','instances':[]}
    report=read_json(path)
    if report.get('model_sha256')!=read_json(bundle/'model.json')['model_sha256'] or report.get('semantics_sha256')!=file_hash(bundle/'semantics/semantics.npz'):
        return {'status':'stale','instances':[]}
    calibration=load_calibration(bundle,report)
    scale=calibration['cm_per_model_unit'] if calibration else None
    paper_path=bundle/'measurements/paper_reference.json'
    paper=read_json(paper_path) if paper_path.is_file() else {}
    return {'status':'calibrated' if calibration else 'unscaled',
        'scale_source':calibration.get('source') if calibration else None,
        'paper_length_cm':calibration.get('paper_length_cm') if calibration else None,
        'paper_width_cm':calibration.get('paper_width_cm') if calibration else None,
        'reference_views':len(calibration.get('inlier_views',[])) if calibration else 0,
        'reason':paper.get('reason') if not calibration else None,
        'instances':[{**entry,'volume_ml':scaled_volume(entry,scale),
            'max_span_cm':entry['max_span_model_units']*scale if scale and entry['max_span_model_units'] else None} for entry in report['instances']],
        'limitation':'可靠高斯中心的凸包包围体积粗估；凹陷、遮挡和缺失表面会产生误差，非视觉称重。'}
