"""Experimental camera/mask-only volume. Not registered in the app or app CLI.

No Gaussian positions, scales, opacity, depth, or baseline volume are read by the
estimator. Saved Gaussian measurements are read only AFTER the estimate to compare.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.optimize import linprog

from .io import read_json, write_json, write_point_cloud_ply
from .schemas import UNKNOWN_ID, load_scene
from .segmentation.vision_llm import select_keyframes


def frustum_bounds(frames, masks, padding=3):
    """Bound the UNION of all foreground, independently of any 3D reconstruction."""
    equations = []
    for frame, labels in zip(frames, masks):
        y, x = np.nonzero((labels > 0) & (labels != UNKNOWN_ID))
        if len(x) < 16: continue
        left, right = max(0, x.min()-padding), min(frame.width-1, x.max()+padding)
        top, bottom = max(0, y.min()-padding), min(frame.height-1, y.max()+padding)
        matrix = np.asarray(frame.K)@np.linalg.inv(np.asarray(frame.camera_to_world))[:3]
        equations.extend([left*matrix[2]-matrix[0], matrix[0]-right*matrix[2],
                          top*matrix[2]-matrix[1], matrix[1]-bottom*matrix[2], -matrix[2]])
    a = np.asarray(equations)
    if len(a) < 15: raise ValueError('至少需要三个有效轮廓视角来限定搜索空间')
    lower, upper = [], []
    for axis in range(3):
        objective = np.eye(3)[axis]
        low = linprog(objective, A_ub=a[:, :3], b_ub=-a[:, 3], bounds=[(None,None)]*3, method='highs')
        high = linprog(-objective, A_ub=a[:, :3], b_ub=-a[:, 3], bounds=[(None,None)]*3, method='highs')
        if not low.success or not high.success:
            raise ValueError('轮廓相机不能限定有限体积，或跨视图边界不一致')
        lower.append(low.fun); upper.append(-high.fun)
    lower, upper = np.array(lower), np.array(upper)
    if (upper-lower <= 0).any(): raise ValueError('轮廓搜索区域退化')
    margin = .02*(upper-lower)
    return lower-margin, upper+margin


def estimate_visual_hull(frames, masks, identifiers, resolution=128, agreement=.95,
                         dilation=1, minimum_support=3, plane=None, save_points=None):
    if not 24 <= resolution <= 384 or not .5 <= agreement <= 1 or not 0 <= dilation <= 5:
        raise ValueError('体素分辨率、轮廓一致性或边界扩张参数无效')
    if len(frames) != len(masks) or len(frames) < 3 or not identifiers:
        raise ValueError('轮廓视角或实例列表无效')
    for frame, labels in zip(frames, masks):
        if labels.shape != (frame.height, frame.width): raise ValueError('掩码与相机分辨率不一致')
    lower, upper = frustum_bounds(frames, masks, padding=max(3,dilation+1))
    step = float((upper-lower).max()/resolution)
    shape = np.ceil((upper-lower)/step).astype(int)
    axes = [lower[i]+(np.arange(shape[i])+.5)*step for i in range(3)]
    positive_masks = []
    for labels in masks:
        positive_masks.append({identifier: binary_dilation(labels == identifier, iterations=dilation)
                               if dilation else labels == identifier for identifier in identifiers})
    volumes = {identifier: {'strict_voxels':0, 'occlusion_aware_voxels':0, 'boundary_voxels':0,
                           'points':[]} for identifier in identifiers}
    # Slice the grid so memory remains bounded at high resolution.
    for start in range(0, shape[0], 8):
        grid = np.meshgrid(axes[0][start:start+8], axes[1], axes[2], indexing='ij')
        points = np.stack(grid, axis=-1).reshape(-1,3)
        permitted = np.ones(len(points),bool)
        if plane is not None:
            permitted &= (points-np.asarray(plane['point']))@np.asarray(plane['normal']) >= 0
        support = np.zeros((len(identifiers),len(points)),np.uint16)
        negative = np.zeros_like(support); strict_negative = np.zeros_like(support)
        for frame, labels, positive in zip(frames,masks,positive_masks):
            inverse = np.linalg.inv(np.asarray(frame.camera_to_world))
            camera = points@inverse[:3,:3].T+inverse[:3,3]
            projected = camera@np.asarray(frame.K).T
            depth = projected[:,2]
            safe = np.where(depth > 1e-8, depth, 1)
            x = np.rint(projected[:,0]/safe).astype(int); y = np.rint(projected[:,1]/safe).astype(int)
            valid = (depth > 1e-8)&(x>=0)&(x<frame.width)&(y>=0)&(y<frame.height)
            vx, vy = np.clip(x,0,frame.width-1), np.clip(y,0,frame.height-1)
            observed = labels[vy,vx]
            for channel, identifier in enumerate(identifiers):
                if np.count_nonzero(labels == identifier) < 16:
                    # A missing detection is not a complete empty silhouette.
                    continue
                hit = positive[identifier][vy,vx]&valid
                support[channel] += hit
                # An unobserved ray or another foreground object can hide this
                # instance. It is not evidence of empty space.
                negative[channel] += valid&(~hit)&(observed == 0)
                strict_negative[channel] += valid&(~hit)&(observed != UNKNOWN_ID)
        for channel, identifier in enumerate(identifiers):
            positive = support[channel]
            eligible = (positive >= minimum_support)&permitted
            strict = eligible&(strict_negative[channel] == 0)
            valid_count = positive+negative[channel]
            relaxed = eligible&(positive >= agreement*np.maximum(valid_count,1))
            entry = volumes[identifier]
            entry['strict_voxels'] += int(strict.sum())
            entry['occlusion_aware_voxels'] += int(relaxed.sum())
            selected = points[relaxed]
            edge = ((selected-lower < step*1.1)|(upper-selected < step*1.1)).any(axis=1)
            entry['boundary_voxels'] += int(edge.sum())
            if save_points: entry['points'].append(selected.astype(np.float32))
    output = []
    for identifier, entry in volumes.items():
        if save_points and entry['points']:
            points = np.concatenate(entry['points'])
            # Limit export size without changing volume counts.
            points = points[::max(1,len(points)//100000)]
            color = np.broadcast_to([(identifier*71)%255,(identifier*139)%255,(identifier*193)%255],points.shape)
            write_point_cloud_ply(Path(save_points)/f'instance_{identifier}.ply',points,color)
        output.append({'instance_id':identifier,
            'volume_strict_units3':entry['strict_voxels']*step**3,
            'volume_occlusion_aware_units3':entry['occlusion_aware_voxels']*step**3,
            'retained_voxels':entry['occlusion_aware_voxels'], 'roi_boundary_voxels':entry['boundary_voxels'],
            'status':'empty' if not entry['occlusion_aware_voxels'] else
                     'roi_clipped' if entry['boundary_voxels'] else 'estimated'})
    return {'instances':output, 'bounds':[lower.tolist(),upper.tolist()], 'voxel_edge_units':step,
            'grid_shape':shape.tolist(),'resolution':resolution,'views':len(frames),'agreement':agreement,
            'dilation_pixels':dilation,'minimum_positive_views':minimum_support,
            'support_plane':plane,'method':'calibrated_mask_voxel_carving_v1'}


def board_reference(frames, camera_root):
    from .board_scale import observe_board,fit_board_tracks,SOURCE
    tracks,_,_=observe_board(frames,camera_root)
    calibration=fit_board_tracks(tracks)
    normal=np.asarray(calibration['board_rotation_model'])[:,2]
    origin=np.asarray(calibration['board_origin_model'])
    centers = np.asarray([f.camera_to_world for f in frames])[:,:3,3]
    if (centers.mean(0)-origin)@normal < 0: normal = -normal
    calibration['support_plane'] = {'point':origin.tolist(),'normal':normal.tolist()}
    calibration['source']=SOURCE
    calibration['status']='calibrated'
    calibration['dimension_source'] = 'known_board_layout_printed_at_100_percent_not_volume_ground_truth'
    return calibration


def fiducial_support_plane(frames, camera_root):
    """Triangulate printed ArUco corners; plane needs no assumed metric size."""
    import cv2
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    detector = cv2.aruco.ArucoDetector(dictionary)
    tracks = {}
    for frame in frames:
        with Image.open(camera_root/frame.image_path) as raw:
            rgb = np.asarray(raw.convert('RGB'))
        corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))
        if ids is None: continue
        projection = np.asarray(frame.K)@np.linalg.inv(np.asarray(frame.camera_to_world))[:3]
        for marker, quad in zip(ids.reshape(-1),corners):
            for corner, pixel in enumerate(quad.reshape(4,2)):
                tracks.setdefault((int(marker),corner),[]).append((projection,pixel,frame.frame_id))
    points, diagnostics = [], []
    for key, track in tracks.items():
        if len(track) < 6: continue
        matrices = np.asarray([x[0] for x in track]);pixels=np.asarray([x[1] for x in track])
        def triangulate(indices):
            equations=[]
            for i in indices:
                p=matrices[i];x,y=pixels[i];equations.extend([x*p[2]-p[0],y*p[2]-p[1]])
            _,_,vh=np.linalg.svd(np.asarray(equations));point=vh[-1]
            return point/point[3] if abs(point[3])>1e-10 else None
        def errors(point):
            image=np.einsum('nij,j->ni',matrices,point)
            residual=np.linalg.norm(image[:,:2]/np.maximum(image[:,2:],1e-10)-pixels,axis=1)
            residual[image[:,2]<=0]=np.inf
            return residual
        from itertools import combinations
        pairs=list(combinations(range(len(track)),2))
        pairs=[pairs[i] for i in np.linspace(0,len(pairs)-1,min(100,len(pairs)),dtype=int)]
        best=None
        for pair in pairs:
            point=triangulate(pair)
            if point is None: continue
            residual=errors(point);inliers=residual<4
            score=(int(inliers.sum()),-float(np.median(residual[inliers])) if inliers.any() else -np.inf)
            if best is None or score>best[0]:best=(score,inliers)
        if best is None or best[1].sum()<6:continue
        point=triangulate(np.flatnonzero(best[1]));residual=errors(point)
        inliers=residual<4
        if inliers.sum()<6:continue
        points.append(point[:3]);diagnostics.append({'marker_id':key[0],'corner':key[1],
            'support_views':int(inliers.sum()),'reprojection_median_px':float(np.median(residual[inliers]))})
    if len(points)<12 or len({x['marker_id'] for x in diagnostics})<3:
        raise ValueError('标记板有效角点不足，无法确认桌面平面')
    points=np.asarray(points);center=points.mean(0)
    _,singular,basis=np.linalg.svd(points-center,full_matrices=False)
    normal=basis[-1];spread=np.linalg.norm(np.ptp(points,axis=0))
    residual=np.abs((points-center)@normal)
    inliers=residual<.015*spread
    if inliers.sum()<12:raise ValueError('标记板角点不共面，无法确认桌面约束')
    center=points[inliers].mean(0);_,_,basis=np.linalg.svd(points[inliers]-center,full_matrices=False)
    normal=basis[-1]
    centers=np.asarray([f.camera_to_world for f in frames])[:,:3,3]
    if (centers.mean(0)-center)@normal<0:normal=-normal
    return {'point':center.tolist(),'normal':normal.tolist(),'source':'triangulated_aruco_board_no_metric_assumption',
            'corners':int(inliers.sum()),'markers':len({d['marker_id'] for d in diagnostics}),
            'plane_rms_model_units':float(np.sqrt(np.mean(((points[inliers]-center)@normal)**2))),
            'corner_diagnostics':diagnostics}


def run_experiment(cameras, segmentation, output, view_counts=(3,6,12,24), resolutions=(96,160), baseline=None):
    cameras, segmentation, output = Path(cameras), Path(segmentation), Path(output)
    if output.exists() and any(output.iterdir()): raise ValueError('实验输出目录必须为空')
    output.mkdir(parents=True,exist_ok=True)
    scene, all_frames = load_scene(cameras)
    data = read_json(segmentation/'segmentation.json')
    instances = read_json(segmentation/'instances.json')
    identifiers = [x['instance_id'] for x in instances['instances'] if 0 < x['instance_id'] < UNKNOWN_ID]
    names = {x['instance_id']:x['name'] for x in instances['instances']}
    usable_frames, excluded_frames = [], []
    for frame in all_frames:
        with Image.open(segmentation/data['masks'][frame.frame_id]) as raw:
            labels = np.asarray(raw)
        y,x = np.nonzero(np.isin(labels, identifiers))
        if len(x) < 100:
            excluded_frames.append({'frame_id':frame.frame_id,'reason':'no_reliable_foreground_silhouette'})
        elif x.min() < 3 or y.min() < 3 or x.max() >= frame.width-3 or y.max() >= frame.height-3:
            excluded_frames.append({'frame_id':frame.frame_id,'reason':'foreground_clipped_by_image_edge'})
        else:
            usable_frames.append(frame)
    if len(usable_frames) < max(view_counts):
        raise ValueError('可用的完整前景视角不足以运行指定数量的对比')
    try:
        calibration = board_reference(all_frames,cameras.parent)
    except (ValueError,OSError) as error:
        calibration = {'status':'unscaled','reason':str(error),'support_plane':None}
    if calibration.get('support_plane') is None:
        try:calibration['support_plane']=fiducial_support_plane(all_frames,cameras.parent)
        except (ValueError,OSError) as error:calibration['support_plane_reason']=str(error)
    write_json(output/'camera_only_calibration.json',calibration)
    runs = []
    for count in view_counts:
        indices = select_keyframes(usable_frames,instances.get('reference_frame'),count)
        frames = [usable_frames[i] for i in indices]
        masks = []
        for frame in frames:
            with Image.open(segmentation/data['masks'][frame.frame_id]) as raw: masks.append(np.asarray(raw).copy())
        for resolution in resolutions:
            started = time.perf_counter()
            result = estimate_visual_hull(frames,masks,identifiers,resolution=resolution,
                plane=calibration.get('support_plane'), save_points=output/f'v{count}_r{resolution}')
            result['elapsed_seconds'] = time.perf_counter()-started
            result['frame_ids'] = [f.frame_id for f in frames]
            for entry in result['instances']:
                entry['name'] = names[entry['instance_id']]
                scale = calibration.get('cm_per_model_unit')
                entry['volume_ml'] = entry['volume_occlusion_aware_units3']*scale**3 if scale else None
            runs.append(result)
            print(f'visual hull: {count} views, {resolution} grid, {result["elapsed_seconds"]:.2f}s',flush=True)
            write_json(output/f'v{count}_r{resolution}.json',result)
    # Baseline is deliberately opened only after all geometry estimates finish.
    original = read_json(baseline) if baseline else None
    comparisons = []
    if original:
        for run in runs:
            for entry in run['instances']:
                old = next((x for x in original['instances'] if x['instance_id']==entry['instance_id']),None)
                if old and old.get('volume_model_units3'):
                    comparisons.append({'views':run['views'],'resolution':run['resolution'],
                        'instance_id':entry['instance_id'],'name':entry['name'],
                        'visual_hull_units3':entry['volume_occlusion_aware_units3'],
                        'gaussian_convex_hull_units3':old['volume_model_units3'],
                        'ratio_to_gaussian':entry['volume_occlusion_aware_units3']/old['volume_model_units3']})
    report = {'camera_file':str(cameras.resolve()),'segmentation':str(segmentation.resolve()),
        'world_units':scene['world_units'],'camera_sha256':hashlib.sha256(cameras.read_bytes()).hexdigest(),
        'gaussian_geometry_used':False,'baseline_used_for_estimation':False,
        'baseline_coordinate_requirement':'baseline must use EXACTLY the supplied camera coordinate system',
        'excluded_frames':excluded_frames,
        'calibration':calibration,'runs':runs,'comparisons':comparisons,
        'limitations':['Camera poses and masks are reused; timing excludes recognition and COLMAP.',
                      'No measured physical volume ground truth; baseline differences are not accuracy.',
                      'Occlusion-aware carving ignores other foreground, which can leave excess volume.',
                      'Missing silhouettes, concavities, hidden bottoms and boundary errors remain.',
                      'Physical mL assumes the PanSense board is flat and printed at its specified size.']}
    write_json(output/'comparison.json',report)
    lines=['# 多视图轮廓体积独立实验','',
        '估计阶段只读取相机、二维实例掩码和已知尺寸的定位板；不读取高斯几何或基线体积。',
        '复用已有相机与掩码，因此耗时不包含识别、分割及 COLMAP；本实验没有实测物理体积真值。','',
        '|视角|网格最长边|实例|轮廓体积（模型单位³）|原高斯凸包体积|比例|',
        '|---:|---:|---|---:|---:|---:|']
    for row in comparisons:
        lines.append(f"|{row['views']}|{row['resolution']}|{row['instance_id']} {row['name']}|{row['visual_hull_units3']:.7g}|{row['gaussian_convex_hull_units3']:.7g}|{row['ratio_to_gaussian']:.3f}|")
    lines += ['', '相机尺度：'+('已通过定位板多视图一致性检查' if calibration.get('cm_per_model_unit') else
                                '未标定；'+calibration.get('reason','')), '',
              '更多视角和更细网格的稳定性应分别检查。与高斯凸包接近不能证明更准确，差距较大也不能仅凭基线判定失败。']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description='独立轮廓体积实验，不接入 App')
    parser.add_argument('--cameras',type=Path,required=True)
    parser.add_argument('--segmentation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path)
    parser.add_argument('--views',type=int,nargs='+',default=[3,6,12,24])
    parser.add_argument('--resolutions',type=int,nargs='+',default=[96,160])
    args=parser.parse_args()
    run_experiment(args.cameras,args.segmentation,args.output,args.views,args.resolutions,args.baseline)


if __name__ == '__main__': main()
