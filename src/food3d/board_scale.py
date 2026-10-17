"""Metric scale from the known PanSense ArUco board and registered cameras."""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
import json
from hashlib import sha256

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .io import read_json, write_json
from .schemas import load_scene
from .visuals import contact_sheet

SOURCE = 'pansense_aruco_board_multiview'


def camera_hash(bundle):
    frames=read_json(Path(bundle)/'scene.json')['frames']
    fields=['frame_id','width','height','K','camera_to_world']
    return sha256(json.dumps([{k:f[k] for k in fields} for f in frames],sort_keys=True).encode()).hexdigest()


def board_hash():
    from pansense.scan_geometry import POSITIONS, MARKER_SIZE
    return sha256(json.dumps({'positions':POSITIONS,'marker_size_m':MARKER_SIZE},sort_keys=True).encode()).hexdigest()


def fit_similarity(source,target):
    source,target=np.asarray(source,float),np.asarray(target,float)
    a,b=source-source.mean(0),target-target.mean(0)
    if len(source)<3 or np.linalg.matrix_rank(a)<2:raise ValueError('定位板角点分布退化')
    u,s,v=np.linalg.svd(b.T@a/len(a));signs=np.ones(3);signs[-1]=np.sign(np.linalg.det(u@v))
    rotation=u@np.diag(signs)@v
    scale=float(np.dot(s,signs)/np.mean(np.sum(a*a,axis=1)))
    if not np.isfinite(scale) or scale<=0:raise ValueError('定位板尺度无效')
    translation=target.mean(0)-scale*rotation@source.mean(0)
    return scale,rotation,translation


def triangulate_track(track,limit_px=4):
    if len(track)<3:return None
    matrices=np.asarray([o['projection'] for o in track]);pixels=np.asarray([o['pixel'] for o in track])
    def fit(indices):
        equations=[]
        for i in indices:
            p=matrices[i];x,y=pixels[i];equations.extend([x*p[2]-p[0],y*p[2]-p[1]])
        _,_,vh=np.linalg.svd(np.asarray(equations));point=vh[-1]
        return point/point[3] if abs(point[3])>1e-10 else None
    def errors(point):
        p=np.einsum('nij,j->ni',matrices,point)
        residual=np.linalg.norm(p[:,:2]/np.maximum(p[:,2:],1e-10)-pixels,axis=1)
        residual[p[:,2]<=0]=np.inf
        return residual
    pairs=list(combinations(range(len(track)),2))
    pairs=[pairs[i] for i in np.linspace(0,len(pairs)-1,min(100,len(pairs)),dtype=int)]
    best=None
    for pair in pairs:
        point=fit(pair)
        if point is None:continue
        residual=errors(point);valid=residual<limit_px
        score=(int(valid.sum()),-float(np.median(residual[valid])) if valid.any() else -np.inf)
        if best is None or score>best[0]:best=(score,valid)
    if best is None or best[1].sum()<3:return None
    point=fit(np.flatnonzero(best[1]))
    residual=errors(point);valid=residual<limit_px
    if valid.sum()<3:return None
    return {'point':point[:3], 'observations':[o for o,ok in zip(track,valid) if ok],
            'reprojection_rms_px':float(np.sqrt(np.mean(residual[valid]**2)))}


def fit_board_tracks(tracks,limit_px=4):
    from pansense.scan_geometry import marker_points, SQUARE_FIRST_ID
    candidates=[]
    for kind in ('square_20cm','a3'):
        keys=[];points=[];triangulated=[]
        for key,track in sorted(tracks.items()):
            if (key[0]>=SQUARE_FIRST_ID)!=(kind=='square_20cm'):continue
            result=triangulate_track(track,limit_px)
            if result is None:continue
            keys.append(key);points.append(result['point']);triangulated.append(result)
        if len(keys)<12 or len({key[0] for key in keys})<3:continue
        source=np.asarray([marker_points(marker)[corner] for marker,corner in keys],float)
        target=np.asarray(points);span=float(np.linalg.norm(np.ptp(target,axis=0)))
        triples=list(combinations(range(len(keys)),3))
        triples=[triples[i] for i in np.linspace(0,len(triples)-1,min(300,len(triples)),dtype=int)]
        best=None
        for triple in triples:
            indices=list(triple)
            try:scale,rotation,translation=fit_similarity(source[indices],target[indices])
            except ValueError:continue
            residual=np.linalg.norm(source@rotation.T*scale+translation-target,axis=1)
            valid=residual<.025*span
            score=(int(valid.sum()),-float(np.median(residual[valid])) if valid.any() else -np.inf)
            if best is None or score>best[0]:best=(score,valid)
        if best is None or best[1].sum()<12:continue
        valid=best[1]
        scale,rotation,translation=fit_similarity(source[valid],target[valid])
        residual=np.linalg.norm(source@rotation.T*scale+translation-target,axis=1)
        valid=residual<.025*span
        if valid.sum()<12 or len({keys[i][0] for i in np.flatnonzero(valid)})<3:continue
        scale,rotation,translation=fit_similarity(source[valid],target[valid])
        # Check held observations against the fitted known board, not just each
        # independently triangulated corner. This also catches layout mismatch.
        projected_errors=[];views=set();corner_records=[];edge_scales=[]
        fitted=source@rotation.T*scale+translation
        for i in np.flatnonzero(valid):
            observations=triangulated[i]['observations']
            for observation in observations:
                p=observation['projection']@np.r_[fitted[i],1]
                if p[2]<=0:continue
                projected_errors.append(float(np.linalg.norm(p[:2]/p[2]-observation['pixel'])))
                views.add(observation['frame_id'])
            corner_records.append({'marker_id':keys[i][0],'corner':keys[i][1],
                'point_model_units':target[i].tolist(),'support_views':len(observations)})
        if len(views)<6:continue
        rms=float(np.sqrt(np.mean(np.square(projected_errors))))
        for marker in {keys[i][0] for i in np.flatnonzero(valid)}:
            positions={corner:i for i,(identifier,corner) in enumerate(keys) if identifier==marker and valid[i]}
            for corner in range(4):
                if corner in positions and (corner+1)%4 in positions:
                    a,b=positions[corner],positions[(corner+1)%4]
                    edge_scales.append(float(np.linalg.norm(target[a]-target[b])/np.linalg.norm(source[a]-source[b])))
        disagreement=float(np.median(np.abs(np.asarray(edge_scales)/scale-1))) if edge_scales else np.inf
        if rms>limit_px or disagreement>.10:continue
        candidate={'board_type':kind,'board_name':'20 × 20 cm 定位板' if kind=='square_20cm' else 'A3 定位板',
            'board_width_cm':20 if kind=='square_20cm' else 42,'board_height_cm':20 if kind=='square_20cm' else 29.7,
            'marker_size_cm':3,'cm_per_model_unit':100/scale,'model_units_per_meter':scale,
            'board_rotation_model':rotation.tolist(),'board_origin_model':translation.tolist(),
            'corners':corner_records,'marker_ids':sorted({keys[i][0] for i in np.flatnonzero(valid)}),
            'inlier_views':sorted(views),'reprojection_rms_px':rms,'reprojection_limit_px':limit_px,
            'edge_scale_median_relative_error':disagreement,
            'board_fit_rms_cm':float(np.sqrt(np.mean(np.sum((fitted[valid]-target[valid])**2,axis=1)))*100/scale)}
        candidates.append(candidate)
    if not candidates:raise ValueError('定位板尺度校验未通过，请拍到至少 3 个标记和 6 个不同视角，并按原尺寸打印定位板')
    return max(candidates,key=lambda x:(len(x['marker_ids']),len(x['corners']),-x['reprojection_rms_px']))


def observe_board(frames,camera_root):
    from pansense.scan_geometry import POSITIONS
    parameters=cv2.aruco.DetectorParameters();parameters.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),parameters)
    tracks={};previews=[];detected=[]
    for frame in frames:
        with Image.open(Path(camera_root)/frame.image_path) as raw:picture=raw.convert('RGB')
        corners,ids,_=detector.detectMarkers(cv2.cvtColor(np.asarray(picture),cv2.COLOR_RGB2GRAY))
        if ids is None:continue
        projection=np.asarray(frame.K)@np.linalg.inv(np.asarray(frame.camera_to_world))[:3]
        view_detections=[]
        preview=picture.copy();draw=ImageDraw.Draw(preview)
        for marker,quad in zip(ids.reshape(-1),corners):
            marker=int(marker)
            if marker not in POSITIONS:continue
            quad=quad.reshape(4,2)
            for corner,pixel in enumerate(quad):
                tracks.setdefault((marker,corner),[]).append({'projection':projection,'pixel':pixel,'frame_id':frame.frame_id})
            view_detections.append({'marker_id':marker,'corners':quad.tolist()})
            xy=[tuple(map(float,p)) for p in quad];draw.line(xy+[xy[0]],fill='lime',width=3)
            draw.text(xy[0],str(marker),fill='red')
        if view_detections:
            detected.append({'frame_id':frame.frame_id,'markers':view_detections})
            if len(previews)<16:
                preview.thumbnail((400,400));previews.append((frame.frame_id,preview))
    return tracks,detected,previews


def calibrate_board(bundle:Path):
    bundle=Path(bundle).resolve();_,frames=load_scene(bundle/'scene.json')
    model=read_json(bundle/'model.json');directory=bundle/'measurements';directory.mkdir(exist_ok=True)
    tracks,detected,previews=observe_board(frames,bundle)
    result={'schema_version':1,'source':SOURCE,'model_sha256':model['model_sha256'],
        'camera_sha256':camera_hash(bundle),'board_sha256':board_hash(),'status':'unavailable',
        'detected_views':len(detected),'total_views':len(frames),'detections':detected,
        'limitations':['Assumes the PanSense board is flat and printed at 100% of its specified dimensions.',
                      'Metric scale checks do not certify physical object volume accuracy.']}
    try:
        result.update(fit_board_tracks(tracks),status='calibrated')
        write_json(directory/'calibration.json',result)
    except ValueError as error:
        result['reason']=str(error)
        previous=directory/'calibration.json'
        if previous.is_file() and read_json(previous).get('source') in {SOURCE,'white_paper_rectangle_multiview'}:
            write_json(previous,{'source':SOURCE,'status':'disabled','reason':str(error)})
    write_json(directory/'board_reference.json',result)
    if previews:contact_sheet(previews,directory/'board_detections.jpg')
    return {k:v for k,v in result.items() if k not in {'detections','corners'}}
