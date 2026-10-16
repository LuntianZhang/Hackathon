"""Camera-only capture coverage from a known planar ArUco board.

Board coordinates are metres, x right, y up on the printed sheet, z up.
No phone orientation, elapsed time, or client-supplied angle is trusted.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

MARKER_SIZE = .03
# Tolerances are px at ~1280 px frames; 30-degree cells tolerate a few degrees.
CALIBRATION_ERROR_PX = 4.0
POSE_ERROR_PX = 7.0
MARKERS = [(x, y) for y in (-.112, 0, .112) for x in (-.175, -.0875, 0, .0875, .175)
           if y != 0 or abs(x) == .175]


# 20 cm square board: eight markers (10 mm margin) around a blank 12 cm centre. Ids 20-27 do
# not collide with the A3 board, so either sheet can be used.
SQUARE_FIRST_ID = 20
SQUARE_MARKERS = {SQUARE_FIRST_ID+i: (x, y) for i, (x, y) in enumerate(
    (x, y) for y in (-.075, 0, .075) for x in (-.075, 0, .075) if (x, y) != (0, 0))}
POSITIONS = {i: p for i, p in enumerate(MARKERS)} | SQUARE_MARKERS


def marker_points(identifier):
    x, y = POSITIONS[identifier]
    s = MARKER_SIZE/2
    return np.array([[x-s,y+s,0],[x+s,y+s,0],[x+s,y-s,0],[x-s,y-s,0]],np.float32)


def square_board_svg():
    """20 cm x 20 cm board in true millimetres (viewBox units = mm)."""
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="200mm" height="200mm" viewBox="0 0 200 200">',
             '<rect width="200" height="200" fill="white"/>',
             '<rect x=".25" y=".25" width="199.5" height="199.5" fill="none" stroke="#888" stroke-width=".5"/>',
             '<rect x="40" y="40" width="120" height="120" fill="none" stroke="#aaa" stroke-dasharray="2 2"/>',
             '<text x="100" y="98" text-anchor="middle" font-size="5">PanSense · PLACE FOOD HERE</text>',
             '<text x="100" y="106" text-anchor="middle" font-size="3.5">20 x 20 cm · PRINT AT 100%</text>']
    for identifier,(x,y) in SQUARE_MARKERS.items():
        cells = cv2.aruco.generateImageMarker(dictionary,identifier,6)
        left,top=100+x*1000-15,100-y*1000-15
        parts.append(f'<rect x="{left-3}" y="{top-3}" width="36" height="36" fill="white"/>')
        for row,col in zip(*np.where(cells==0)):
            parts.append(f'<rect x="{left+col*5}" y="{top+row*5}" width="5.01" height="5.01" fill="black"/>')
    parts.append('</svg>')
    return ''.join(parts)


def square_board_png(dpi=300):
    """Same board as a PNG with exact physical size metadata."""
    size = round(200/25.4*dpi)
    scale = size/200
    image = np.full((size,size),255,np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    side = round(30*scale)
    for identifier,(x,y) in SQUARE_MARKERS.items():
        marker = cv2.aruco.generateImageMarker(dictionary,identifier,side,borderBits=1)
        left,top = round((100+x*1000-15)*scale),round((100-y*1000-15)*scale)
        image[top:top+side,left:left+side] = marker
    cv2.rectangle(image,(0,0),(size-1,size-1),136,2)
    cv2.putText(image,'PanSense  20 x 20 cm  PRINT AT 100%',(round(40*scale),round(100*scale)),
                cv2.FONT_HERSHEY_SIMPLEX,size/1800,60,max(1,size//900),cv2.LINE_AA)
    ok,encoded = cv2.imencode('.png',image)
    data = bytearray(encoded.tobytes())
    ppm = round(dpi/.0254)
    chunk = b'pHYs'+ppm.to_bytes(4,'big')+ppm.to_bytes(4,'big')+b''
    import zlib
    record = len(chunk[4:]).to_bytes(4,'big')+chunk+zlib.crc32(chunk).to_bytes(4,'big')
    data[33:33] = record   # right after IHDR
    return bytes(data)


def board_svg():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="420mm" height="297mm" viewBox="0 0 420 297">',
             '<rect width="420" height="297" fill="white"/>',
             '<circle cx="210" cy="148.5" r="105" fill="none" stroke="#aaa" stroke-dasharray="2 2"/>',
             '<text x="210" y="145" text-anchor="middle" font-size="6">PanSense · PLACE PLATE HERE</text>',
             '<text x="210" y="155" text-anchor="middle" font-size="4">A3 LANDSCAPE · PRINT AT 100% · DO NOT SCALE</text>']
    for identifier,(x,y) in enumerate(MARKERS):
        cells = cv2.aruco.generateImageMarker(dictionary,identifier,6)
        left,top=210+x*1000-15,148.5-y*1000-15
        parts.append(f'<rect x="{left-3}" y="{top-3}" width="36" height="36" fill="white"/>')
        for row,col in zip(*np.where(cells==0)):
            parts.append(f'<rect x="{left+col*5}" y="{top+row*5}" width="5.01" height="5.01" fill="black"/>')
        parts.append(f'<text x="{left+15}" y="{top+35}" text-anchor="middle" font-size="3">{identifier}</text>')
    parts.extend(['<path d="M10 285h100m-100-2v4m100-4v4" stroke="black" fill="none"/>',
                  '<text x="60" y="282" font-size="4" text-anchor="middle">100 mm</text></svg>'])
    return ''.join(parts)


def image_observation(image):
    rgb=np.asarray(image)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    # Check food-area sharpness independently from high-contrast board markers.
    h,w=gray.shape;crop=gray[h//4:3*h//4,w//4:3*w//4]
    sharpness=float(cv2.Laplacian(crop,cv2.CV_64F).var())
    dark=float(np.mean(crop<12));bright=float(np.mean(crop>245))
    quality='blur' if sharpness<25 else 'exposure' if dark>.7 or bright>.85 else 'ok'
    detector=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners,identifiers,_=detector.detectMarkers(gray)
    objects=[];pixels=[];ids=[]
    for index,identifier in enumerate([] if identifiers is None else identifiers.flatten()):
        if int(identifier) not in POSITIONS: continue
        objects.extend(marker_points(int(identifier)).tolist())
        pixels.extend(corners[index].reshape(4,2).tolist());ids.append(int(identifier))
    return {'width':w,'height':h,'sharpness':round(sharpness,2),'quality':quality,
            'marker_ids':ids,'objects':objects,'pixels':pixels}


def calibrate(observations):
    """Estimate intrinsics from diverse saved board views, never a guessed FOV."""
    good=[o for o in observations if o['quality']=='ok' and len(o['marker_ids'])>=3]
    selected=[];homographies=[]
    for observation in good:
        objects=np.array(observation['objects'],np.float32)
        pixels=np.array(observation['pixels'],np.float32)
        h,_=cv2.findHomography(objects[:,:2],pixels,cv2.RANSAC,2.5)
        if h is None: continue
        # Sample centre and basis points in a common board frame. Repeated
        # frames cannot manufacture a diverse calibration set.
        descriptor=cv2.perspectiveTransform(np.array([[[0,0],[.1,0],[0,.1]]],np.float32),h)[0].flatten()
        if any(np.linalg.norm(descriptor-previous)<35 for previous in homographies): continue
        selected.append(observation);homographies.append(descriptor)
        if len(selected)>=24: break
    if len(selected)<8: return None
    width,height=selected[0]['width'],selected[0]['height']
    selected=[o for o in selected if (o['width'],o['height'])==(width,height)]
    if len(selected)<8: return None
    flags=(cv2.CALIB_FIX_ASPECT_RATIO|cv2.CALIB_ZERO_TANGENT_DIST|
           cv2.CALIB_FIX_K1|cv2.CALIB_FIX_K2|cv2.CALIB_FIX_K3)
    # Phone video frames fit a pinhole model only to a few pixels (autofocus
    # breathing, slightly bent paper), so drop the worst-fitting views.
    try:
        while True:
            error,k,dist,rvecs,tvecs=cv2.calibrateCamera(
                [np.array(o['objects'],np.float32) for o in selected],
                [np.array(o['pixels'],np.float32) for o in selected],(width,height),None,None,flags=flags)
            if error<=CALIBRATION_ERROR_PX or len(selected)<=8: break
            residuals=[]
            for o,r,t in zip(selected,rvecs,tvecs):
                projected,_=cv2.projectPoints(np.array(o['objects'],np.float32),r,t,k,dist)
                residuals.append(float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-np.array(o['pixels']))**2,axis=1)))))
            selected.pop(int(np.argmax(residuals)))
    except cv2.error:
        return None
    normals=[cv2.Rodrigues(r)[0][:,2] for r in rvecs]
    diversity=max(math.degrees(math.acos(float(np.clip(np.dot(a,b),-1,1)))) for a in normals for b in normals)
    if not np.isfinite(k).all() or not np.isfinite(error) or error>CALIBRATION_ERROR_PX or diversity<12:
        return None
    if not .35*width<k[0,0]<4*width: return None
    if not .35*width<k[0,2]<.65*width or not .35*height<k[1,2]<.65*height: return None
    return {'K':k.tolist(),'distortion':dist.flatten().tolist(),'width':width,'height':height,
            'reprojection_error':float(error),'views':len(selected),'tilt_range_deg':diversity}


def estimate_pose(observation, calibration):
    if not calibration or len(observation['marker_ids'])<2: return None
    if (observation['width'],observation['height'])!=(calibration['width'],calibration['height']): return None
    objects=np.array(observation['objects'],np.float64).reshape(-1,4,3)
    pixels=np.array(observation['pixels'],np.float64).reshape(-1,4,2)
    k=np.array(calibration['K'],dtype=np.float64);dist=np.array(calibration['distortion'],dtype=np.float64)
    keep=list(range(len(objects)))
    # Planar solver on whole markers; drop the worst marker until the fit is good.
    while True:
        flat_o=objects[keep].reshape(-1,3);flat_p=pixels[keep].reshape(-1,2)
        try:
            ok,r,t=cv2.solvePnP(flat_o,flat_p,k,dist,flags=cv2.SOLVEPNP_IPPE)
            if not ok: return None
            r,t=cv2.solvePnPRefineLM(flat_o,flat_p,k,dist,r,t)
            projected,_=cv2.projectPoints(flat_o,r,t,k,dist)
        except cv2.error:
            return None
        gaps=projected.reshape(-1,2)-flat_p
        error=float(np.sqrt(np.mean(np.sum(gaps**2,axis=1))))
        if error<=POSE_ERROR_PX: break
        if len(keep)<=2: return None
        keep.pop(int(np.argmax(np.sqrt(np.sum(gaps**2,axis=1)).reshape(-1,4).mean(axis=1))))
    rotation=cv2.Rodrigues(r)[0];centre=(-rotation.T@t).flatten()
    distance=float(np.linalg.norm(centre))
    if not np.isfinite(centre).all() or centre[2]<=0 or not .1<distance<1.8: return None
    target,_=cv2.projectPoints(np.zeros((1,3)),r,t,k,dist);u,v=target.flatten()
    if not .15*observation['width']<u<.85*observation['width'] or not .15*observation['height']<v<.85*observation['height']: return None
    # Look towards board centre, rather than merely keeping a corner visible.
    return {'centre':centre.tolist(),'azimuth':math.degrees(math.atan2(centre[0],-centre[1]))%360,
            'elevation':math.degrees(math.atan2(centre[2],np.linalg.norm(centre[:2]))),
            'reprojection_error':error,'distance':distance}


def coverage(frames, calibration,reference_azimuth=None):
    # Recompute from surviving evidence after every upload/deletion/calibration.
    reference=next((f.get('pose') for f in frames if f.get('pose') and f['observation']['quality']=='ok'),None)
    if reference_azimuth is None and reference: reference_azimuth=reference['azimuth']
    cells=[{'index':i,'level':i//12,'direction':i%12,'status':'missing','frame_ids':[],'valid_frames':0} for i in range(24)]
    current=None;pending=[];accepted=[];counts={}
    for frame in frames:
        pose=frame.get('pose');quality=frame['observation']['quality']
        current=None
        if frame.get('duplicate_image'):
            frame['duplicate_view']=True;frame['sector']=None;continue
        manual=frame.get('manual_sector')
        if not pose and manual is not None and 0<=manual<24:
            # Hand-labelled direction: shown separately, never counted as measured.
            frame['sector']=manual;cell=cells[manual];cell['frame_ids'].append(frame['id'])
            if quality!='ok':
                if cell['status'] not in ('covered','manual'): cell['status']='retake'
            elif cell['status'] not in ('covered','partial'): cell['status']='manual'
            continue
        if not pose or not reference:
            pending.append(frame['id']);frame['sector']=None;continue
        relative=(pose['azimuth']-reference_azimuth+15)%360
        elev=pose['elevation'];level=0 if 15<=elev<40 else 1 if 40<=elev<=90 else None
        if level is None:
            pending.append(frame['id']);frame['sector']=None;continue
        sector=level*12+int(relative//30);frame['sector']=sector;cell=cells[sector]
        cell['frame_ids'].append(frame['id']);current={'sector':sector,'elevation':round(elev,1),'azimuth':round(relative-15,1),'frame_id':frame['id']}
        if quality!='ok':
            if cell['status']!='covered': cell['status']='retake'
            continue
        previous=counts.setdefault(sector,[])
        if any(abs((pose['azimuth']-p['azimuth']+180)%360-180)<4 or np.linalg.norm(np.array(pose['centre'])-p['centre'])<.015 for p in previous):
            frame['duplicate_view']=True;continue
        frame['duplicate_view']=False;previous.append(pose);accepted.append(frame['id'])
        cell['valid_frames']=len(previous);cell['status']='covered' if len(previous)>=2 else 'partial'
    missing=[c['index'] for c in cells if c['status']!='covered']
    manual_cells=sum(c['status']=='manual' for c in cells)
    return {'cells':cells,'covered':24-len(missing),'manual':manual_cells,'total':24,'missing':missing,'current':current,
            'pending_frames':pending,'accepted_frame_ids':accepted,
            'calibration':{'ready':bool(calibration),'views':calibration['views'] if calibration else 0},
            'basis':'saved_board_camera_poses','geometry_verified':False}
