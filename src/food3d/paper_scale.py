"""Metric scale from a stationary white rectangle and calibrated camera poses.

The rectangle center has no corner identity ambiguity under a 180 degree turn.
Each PnP estimate satisfies P_cm = scale*C_model + R_c2w*t_paper_cm.
We robustly fit this common center across views, then independently triangulate
the aligned corners and check both known edge lengths before accepting scale.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from hashlib import sha256
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .io import read_json, write_json
from .schemas import load_scene
from .visuals import contact_sheet

SOURCE = 'white_paper_rectangle_multiview'


def camera_hash(bundle):
    frames=read_json(Path(bundle)/'scene.json')['frames']
    fields=['frame_id','width','height','K','camera_to_world']
    return sha256(json.dumps([{k:f[k] for k in fields} for f in frames],sort_keys=True).encode()).hexdigest()


def detect_paper(rgb):
    hsv=cv2.cvtColor(np.asarray(rgb),cv2.COLOR_RGB2HSV)
    mask=((hsv[:,:,1]<70)&(hsv[:,:,2]>140)).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    h,w=mask.shape;candidates=[]
    for contour in contours:
        area=cv2.contourArea(contour)
        if not .0002*w*h<area<.12*w*h: continue
        hull=cv2.convexHull(contour)
        quad=cv2.approxPolyDP(hull,.025*cv2.arcLength(hull,True),True)
        if len(quad)!=4 or not cv2.isContourConvex(quad): continue
        points=quad[:,0].astype(np.float64)
        if np.any(points<3) or np.any(points[:,0]>w-4) or np.any(points[:,1]>h-4): continue
        center=points.mean(0);angle=np.arctan2(points[:,1]-center[1],points[:,0]-center[0])
        points=points[np.argsort(angle)]
        sides=np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1)
        if sides[0]+sides[2]<sides[1]+sides[3]: points=np.roll(points,-1,axis=0);sides=np.roll(sides,-1)
        ratio=(sides[0]+sides[2])/(sides[1]+sides[3])
        if not 2<ratio<12 or sides.min()<5 or sides.max()<30: continue
        if area/max(cv2.contourArea(points.astype(np.float32)),1)<.85: continue
        candidates.append({'corners':points,'area':area,'aspect_pixels':float(ratio)})
    return sorted(candidates,key=lambda d:d['area'],reverse=True)


def rectangle(length_cm,width_cm):
    if not np.isfinite([length_cm,width_cm]).all() or not length_cm>width_cm>0:
        raise ValueError('纸片长边必须大于短边，两者均为正数，单位厘米')
    return np.array([[-length_cm/2,-width_cm/2,0],[length_cm/2,-width_cm/2,0],
                     [length_cm/2,width_cm/2,0],[-length_cm/2,width_cm/2,0]],np.float64)


def observation(frame,corners,length_cm=10,width_cm=2):
    objects=rectangle(length_cm,width_cm);k=np.asarray(frame.K,np.float64)
    count,rvecs,tvecs,_=cv2.solvePnPGeneric(objects,np.asarray(corners,np.float64),k,None,flags=cv2.SOLVEPNP_IPPE)
    # IPPE has numerically ambiguous symmetric frontal cases. Keep the standard
    # iterative planar solution as a second candidate and still check its error.
    ok,r,t=cv2.solvePnP(objects,np.asarray(corners,np.float64),k,None,flags=cv2.SOLVEPNP_ITERATIVE)
    if ok: rvecs=(*rvecs,r);tvecs=(*tvecs,t)
    choices=[];pose=np.asarray(frame.camera_to_world)
    for r,t in zip(rvecs,tvecs):
        rotation=cv2.Rodrigues(r)[0];camera=objects@rotation.T+t.reshape(3)
        if not np.isfinite(camera).all() or camera[:,2].min()<=0: continue
        projected=cv2.projectPoints(objects,r,t,k,None)[0][:,0]
        error=float(np.sqrt(np.mean(np.sum((projected-corners)**2,axis=1))))
        if error>3: continue
        choices.append({'frame':frame,'corners':np.asarray(corners),'camera_center':pose[:3,3],
            'offset_cm':pose[:3,:3]@t.reshape(3),'rotation_world':pose[:3,:3]@rotation,
            'pnp_error_px':error})
    return min(choices,key=lambda v:v['pnp_error_px']) if choices else None


def fit_scale(observations,length_cm=10,width_cm=2,min_views=6):
    rectangle(length_cm,width_cm)
    if len(observations)<min_views: raise ValueError('完整纸片的有效视角不足，至少需要 6 个可用视角')
    c=np.array([o['camera_center'] for o in observations]);v=np.array([o['offset_cm'] for o in observations])
    def fit(indices):
        a=c[indices]-c[indices].mean(0);b=v[indices]-v[indices].mean(0)
        denominator=float(np.sum(a*a))
        scale=-float(np.sum(a*b))/denominator if denominator>1e-12 else -1
        return scale,np.mean(scale*c[indices]+v[indices],axis=0)
    # Deterministic RANSAC on camera pairs; false white regions cannot define
    # physical scale unless they are mutually consistent across moving cameras.
    pairs=list(combinations(range(len(c)),2))
    if len(pairs)>800: pairs=[pairs[i] for i in np.linspace(0,len(pairs)-1,800,dtype=int)]
    best=None
    for pair in pairs:
        scale,center=fit(list(pair))
        if scale<=0: continue
        residual=np.linalg.norm(scale*c+v-center,axis=1);inliers=residual<.12*length_cm
        if not inliers.any(): continue
        score=(int(inliers.sum()),-float(np.median(residual[inliers])))
        if best is None or score>best[0]: best=(score,inliers)
    if best is None or best[1].sum()<min_views: raise ValueError('纸片在多视角中不一致，无法可靠标定')
    indices=np.flatnonzero(best[1]);scale,center=fit(indices)
    residual=np.linalg.norm(scale*c+v-center,axis=1)
    indices=np.flatnonzero(residual<.12*length_cm)
    if len(indices)<min_views: raise ValueError('通过一致性校验的纸片视角不足')
    scale,center=fit(indices)
    residual=np.linalg.norm(scale*c+v-center,axis=1)
    if np.max(np.linalg.norm(c[indices]-c[indices].mean(0),axis=1))*scale<length_cm*.25:
        raise ValueError('相机移动基线不足，无法从视频确认纸片尺度')
    objects=rectangle(length_cm,width_cm)
    anchor=observations[indices[np.argmin([observations[i]['pnp_error_px'] for i in indices])]]
    reference=objects@anchor['rotation_world'].T
    aligned=[]
    for i in indices:
        o=observations[i];world=objects@o['rotation_world'].T
        # Long and short edges keep their identities. Only rectangle symmetries
        # are permitted; no arbitrary corner correspondence is invented.
        permutations=[[0,1,2,3],[2,3,0,1],[1,0,3,2],[3,2,1,0]]
        order=min(permutations,key=lambda p:float(np.sum((world[p]-reference)**2)))
        aligned.append((o,o['corners'][order]))
    points=[]
    for corner in range(4):
        equations=[]
        for o,pixels in aligned:
            frame=o['frame'];projection=np.asarray(frame.K)@np.linalg.inv(np.asarray(frame.camera_to_world))[:3]
            x,y=pixels[corner];equations.extend([x*projection[2]-projection[0],y*projection[2]-projection[1]])
        _,_,vh=np.linalg.svd(np.asarray(equations));point=vh[-1]
        if abs(point[3])<1e-10: raise ValueError('纸片角点无法三角化')
        points.append(point[:3]/point[3])
    points=np.array(points);edges=np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1)
    long=float(np.mean(edges[[0,2]]));short=float(np.mean(edges[[1,3]]))
    long_scale=length_cm/long;short_scale=width_cm/short
    errors=[]
    for o,pixels in aligned:
        frame=o['frame'];p=points@np.linalg.inv(np.asarray(frame.camera_to_world))[:3,:3].T+np.linalg.inv(np.asarray(frame.camera_to_world))[:3,3]
        if p[:,2].min()<=0: raise ValueError('三角化纸片位于相机后方')
        image=p@np.asarray(frame.K).T;errors.extend(np.linalg.norm(image[:,:2]/image[:,2:]-pixels,axis=1))
    reprojection=float(np.sqrt(np.mean(np.square(errors))))
    disagreement=abs(long_scale-short_scale)/long_scale
    # Hand-cut edges and video blur add corner noise. At ~960 px long edge,
    # accept up to 4 px RMS while independently enforcing both metric edges.
    pixel_limit=max(3.0,4.0*np.median([max(o['frame'].width,o['frame'].height) for o,_ in aligned])/960)
    if disagreement>.18 or abs(long_scale-scale)/scale>.18 or reprojection>pixel_limit:
        raise ValueError(f'纸片校验未通过：长短边尺度差 {disagreement:.1%}，中心尺度差 {abs(long_scale-scale)/scale:.1%}，重投影 {reprojection:.2f}px；暂不输出物理体积')
    return {'cm_per_model_unit':float(long_scale),'long_edge_cm_check':long*long_scale,
            'short_edge_cm_check':short*long_scale,'edge_scale_disagreement':disagreement,
            'center_fit_scale':float(scale),'reprojection_rms_px':reprojection,
            'reprojection_limit_px':float(pixel_limit),
            'center_residual_median_cm':float(np.median(residual[indices])),
            'inlier_views':[observations[i]['frame'].frame_id for i in indices],
            'corners_model_units':points.tolist(),'detected_views':len(observations)}


def calibrate_paper(bundle:Path,length_cm=10,width_cm=2):
    bundle=bundle.resolve();rectangle(length_cm,width_cm)
    metadata,frames=load_scene(bundle/'scene.json');model=read_json(bundle/'model.json')
    observations=[];debug=[];detections=[]
    for frame in frames:
        with Image.open(bundle/frame.image_path) as raw: picture=raw.convert('RGB')
        candidates=detect_paper(np.asarray(picture));accepted=None
        for candidate in candidates:
            accepted=observation(frame,candidate['corners'],length_cm,width_cm)
            if accepted: break
        if accepted:
            observations.append(accepted);detections.append({'frame_id':frame.frame_id,'corners':accepted['corners'].tolist(),'pnp_error_px':accepted['pnp_error_px']})
            if len(debug)<24:
                preview=picture.copy();draw=ImageDraw.Draw(preview);xy=[tuple(p) for p in accepted['corners']];draw.line(xy+[xy[0]],fill='lime',width=3)
                debug.append((frame.frame_id,preview))
    directory=bundle/'measurements';directory.mkdir(exist_ok=True)
    result={'schema_version':1,'source':SOURCE,'model_sha256':model['model_sha256'],
            'camera_sha256':camera_hash(bundle),'paper_length_cm':length_cm,'paper_width_cm':width_cm,
            'detections':detections,'total_views':len(frames),'status':'unavailable',
            'limitations':['Paper must be flat, stationary, fully visible in several views and exactly measured',
                           'Scale consistency is not physical volume ground truth; convex hull errors remain']}
    try:
        result.update(fit_scale(observations,length_cm,width_cm),status='calibrated')
        write_json(directory/'calibration.json',result)
    except ValueError as error:
        result['reason']=str(error)
        previous=directory/'calibration.json'
        if previous.is_file() and read_json(previous).get('source')==SOURCE: previous.unlink()
    write_json(directory/'paper_reference.json',result)
    if debug: contact_sheet(debug,directory/'paper_detections.jpg')
    return result
