"""Image-only positioning-board checks for vision-model object proposals."""
from __future__ import annotations

import numpy as np


def board_polygon(image):
    """Recover the printed sheet boundary from its known marker layout, no pose."""
    try:
        import cv2
        from pansense.scan_geometry import POSITIONS,marker_points
    except ImportError:
        return None
    params=cv2.aruco.DetectorParameters();params.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),params)
    rgb=np.asarray(image.convert('RGB'))
    try:corners,identifiers,_=detector.detectMarkers(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY))
    except cv2.error:return None
    detections={int(i):c.reshape(4,2) for i,c in zip([] if identifiers is None else identifiers.flatten(),corners)
                if int(i) in POSITIONS}
    candidates=[]
    for square,width,height in ((True,.2,.2),(False,.42,.297)):
        ids=[i for i in detections if (i>=20)==square]
        if len(ids)<3:continue
        source=np.concatenate([marker_points(i)[:,:2] for i in ids]).astype(np.float64)
        target=np.concatenate([detections[i] for i in ids]).astype(np.float64)
        matrix,inliers=cv2.findHomography(source,target,cv2.RANSAC,3)
        if matrix is None or inliers is None or int(inliers.sum())<max(8,len(source)//2):continue
        boundary=np.array([[[-width/2,height/2],[width/2,height/2],
                             [width/2,-height/2],[-width/2,-height/2]]],np.float64)
        polygon=cv2.perspectiveTransform(boundary,matrix)[0].astype(np.float32)
        if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon):continue
        if cv2.contourArea(polygon)<image.width*image.height*.0025:continue
        candidates.append((int(inliers.sum()),{'layout':'square_20cm' if square else 'a3',
            'polygon_pixels':polygon.tolist(),'marker_ids':ids,'inlier_corners':int(inliers.sum())}))
    return max(candidates,key=lambda x:x[0])[1] if candidates else None


def filter_to_board(objects,views):
    """Drop outside-board observations; retain complete silhouettes of kept objects.

    Box centres are a conservative proposal check, not a 3D contact-point test.
    Views without a verified marker layout still use the model's board-only prompt.
    """
    import cv2
    regions={fid:board_polygon(image) for fid,image in views}
    sizes={fid:image.size for fid,image in views}
    kept=[];decisions=[]
    for obj in objects:
        inside=[];outside=[];unverified=[]
        for observation in obj['observations']:
            fid=observation['frame_id'];region=regions[fid]
            if region is None:
                unverified.append(observation);continue
            width,height=sizes[fid];left,top,right,bottom=observation['box']
            centre=((left+right)*width/2,(top+bottom)*height/2)
            polygon=np.asarray(region['polygon_pixels'],np.float32)
            (inside if cv2.pointPolygonTest(polygon,centre,False)>=0 else outside).append(observation)
        reject=len(outside)>len(inside)
        observations=inside+unverified
        if not reject and observations:kept.append({**obj,'observations':observations})
        decisions.append({'instance_id':obj['instance_id'],'name':obj['name'],
            'retained':not reject and bool(observations),'inside_views':len(inside),'outside_views':len(outside),
            'unverified_views':len(unverified)})
    return kept,{'scope':'white_positioning_board','status':'checked' if any(regions.values()) else 'prompt_only',
        'frames':regions,'objects':decisions,
        'unverified_view_policy':'model must identify the white board visually and omit objects outside it',
        'pixel_policy':'keep full object silhouette; do not clip elevated surfaces to board polygon'}


def scoped_result(result,views):
    objects,scope=filter_to_board(result['objects'],views)
    if not objects:raise ValueError('白色定位板内未找到可靠对象，请确认定位板及板内物体清晰可见')
    return {**result,'objects':objects,'recognition_scope':'white_positioning_board','board_scope':scope}
