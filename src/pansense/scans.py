"""Persistent, paired phone capture sessions; only saved evidence counts."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path
import re
import secrets
import shutil
import threading
from uuid import uuid4

import cv2
from PIL import Image, ImageOps, UnidentifiedImageError

from . import scan_geometry as geometry
from .store import ConflictError, now


def qr_data(value):
    import base64
    qr=cv2.QRCodeEncoder_create().encode(value)
    qr=cv2.copyMakeBorder(qr,4,4,4,4,cv2.BORDER_CONSTANT,value=255)
    qr=cv2.resize(qr,None,fx=6,fy=6,interpolation=cv2.INTER_NEAREST)
    ok,png=cv2.imencode('.png',qr)
    if not ok: raise ValueError('无法生成连接二维码')
    return 'data:image/png;base64,'+base64.b64encode(png).decode('ascii')


class Scans:
    def __init__(self,store,jobs,project):
        self.store,self.jobs,self.project=store,jobs,project
        self.root=store.root/'scans';self.root.mkdir(exist_ok=True)
        self.lock=threading.RLock()
        self.public_url=None;self.ca_path=None;self.certificate_fingerprint=None
        from .mobile_https import MobileGateway
        self.gateway=MobileGateway(self)
        jobs.verify_scan=self.verify_scene
        with store.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS scans (id TEXT PRIMARY KEY, value TEXT NOT NULL)')

    def get(self,identifier):
        if not re.fullmatch(r'[a-f0-9]{32}',identifier): raise KeyError('扫描记录不存在')
        with self.store.connection() as db:
            row=db.execute('SELECT value FROM scans WHERE id=?',(identifier,)).fetchone()
        if row is None: raise KeyError('扫描记录不存在')
        return json.loads(row['value'])

    def save(self,value):
        value['updated_at']=now();value['revision']+=1
        with self.store.connection() as db:
            db.execute('INSERT OR REPLACE INTO scans VALUES(?,?)',(value['id'],json.dumps(value,ensure_ascii=False)))
        return value

    def list(self):
        with self.store.connection() as db: rows=db.execute('SELECT value FROM scans').fetchall()
        return [self.public(v) for v in sorted((json.loads(r['value']) for r in rows),key=lambda v:v['updated_at'],reverse=True)]

    def public(self,value):
        report=value.get('coverage') or geometry.coverage([],None)
        return {k:value.get(k) for k in ('id','title','status','revision','created_at','updated_at','meal_id','verification')}|{
            'frame_count':len(value['frames']),'good_frames':sum(f['observation']['quality']=='ok' for f in value['frames']),
            'coverage':report,'frames':[{'id':f['id'],'quality':f['observation']['quality'],
                'sharpness':f['observation']['sharpness'],'markers':len(f['observation']['marker_ids']),
                'sector':f.get('sector'),'duplicate_view':f.get('duplicate_view',False),
                'located':bool(f.get('pose')),'manual':f.get('manual_sector') is not None and not f.get('pose'),'url':f'/api/scans/{value["id"]}/frames/{f["id"]}'} for f in value['frames']]}

    def create(self,title):
        with self.lock:
            value={'id':uuid4().hex,'revision':0,'title':title,'created_at':now(),'updated_at':now(),
                   'status':'paused','frames':[],'calibration':None,'token_hash':None,'meal_id':None,
                   'verification':None}
            self.save(value)
            return self.pair(value['id'])

    def pair(self,identifier):
        with self.lock:
            value=self.get(identifier);token=secrets.token_urlsafe(32)
            value['token_hash']=sha256(token.encode()).hexdigest();self.save(value)
            url=f'{self.public_url}/capture.html#{identifier}/{token}' if self.public_url else None
            return {'scan':self.public(value),'pairing':{'url':url,'qr':qr_data(url) if url else None,
                    'local_url':f'/capture.html#{identifier}/{token}'}}

    def authenticate(self,identifier,token):
        value=self.get(identifier)
        expected=value.get('token_hash')
        if not expected or not secrets.compare_digest(expected,sha256(token.encode()).hexdigest()):
            return False
        return True

    def recalculate(self,value):
        # Incremental: a frame's pose depends only on its own markers and the
        # calibration, so solve each frame once and reuse it afterwards.
        if value['calibration'] is None:
            # Calibration is costly; retry only when a new usable board view exists.
            usable=sum(f['observation']['quality']=='ok' and len(f['observation']['marker_ids'])>=3 for f in value['frames'])
            if usable>=8 and usable!=value.get('calibration_attempt'):
                value['calibration_attempt']=usable
                value['calibration']=geometry.calibrate([f['observation'] for f in value['frames']])
                if value['calibration']:
                    for frame in value['frames']: frame.pop('pose',None)
        for frame in value['frames']:
            if 'pose' not in frame: frame['pose']=geometry.estimate_pose(frame['observation'],value['calibration'])
        if value.get('origin_azimuth') is None:
            first=next((f['pose'] for f in value['frames'] if f.get('pose') and f['observation']['quality']=='ok'),None)
            if first: value['origin_azimuth']=first['azimuth']
        value['coverage']=geometry.coverage(value['frames'],value['calibration'],value.get('origin_azimuth'))

    def update(self,identifier,status,revision):
        with self.lock:
            value=self.get(identifier)
            if value['revision']!=revision: raise ConflictError('扫描状态已变化，请刷新后重试')
            if value.get('meal_id') and status=='capturing': raise ValueError('此扫描已送入重建，请创建新的扫描')
            value['status']=status;return self.public(self.save(value))

    def add_frame(self,identifier,frame_id,data,timestamp,manual_sector=None):
        if manual_sector is not None and not 0<=manual_sector<24: raise ValueError('手动方向编号无效')
        if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',frame_id): raise ValueError('照片编号格式不正确')
        if not math.isfinite(timestamp) or timestamp<0: raise ValueError('照片时间戳无效')
        checksum=sha256(data).hexdigest()
        with self.lock:
            value=self.get(identifier)
            existing=next((f for f in value['frames'] if f['id']==frame_id),None)
            if existing:
                if existing['sha256']!=checksum: raise ConflictError('照片编号已被其他图像使用')
                return self.public(value)
            if value['status']!='capturing': raise ConflictError('扫描已暂停或结束，请先继续扫描')
            if len(value['frames'])>=360: raise ValueError('单次扫描最多保存 360 张照片，请结束本次扫描')
            try:
                with Image.open(BytesIO(data)) as raw:
                    if raw.width*raw.height>16_000_000: raise ValueError('照片像素过高，请降低采集分辨率')
                    image=ImageOps.exif_transpose(raw).convert('RGB')
                    image.thumbnail((1600,1600))
            except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as error:
                raise ValueError('无法读取采集照片') from error
            if min(image.size)<240: raise ValueError('照片分辨率过低')
            if value['frames']:
                previous=value['frames'][0]['observation']
                if image.size!=(previous['width'],previous['height']):
                    raise ValueError('相机尺寸发生变化，请恢复原镜头与方向，或创建新的扫描')
            observation=geometry.image_observation(image)
            duplicate=any(f['sha256']==checksum for f in value['frames'])
            folder=self.root/identifier/'frames';folder.mkdir(parents=True,exist_ok=True)
            path=folder/f'{frame_id}.jpg';image.save(path,quality=94)
            value['frames'].append({'id':frame_id,'sha256':checksum,'timestamp':timestamp,
                'saved_at':now(),'observation':observation,'duplicate_view':duplicate,'duplicate_image':duplicate,
                'manual_sector':manual_sector})
            try:
                self.recalculate(value);self.save(value)
            except Exception:
                path.unlink(missing_ok=True);raise
            return self.public(value)

    def frame_path(self,identifier,frame_id):
        value=self.get(identifier)
        if not any(f['id']==frame_id for f in value['frames']): raise KeyError('照片不存在')
        return self.root/identifier/'frames'/f'{frame_id}.jpg'

    def remove_frame(self,identifier,frame_id,revision):
        with self.lock:
            value=self.get(identifier)
            if value['revision']!=revision: raise ConflictError('照片已变化，请刷新后重试')
            if value.get('meal_id'): raise ValueError('照片已送入重建，无法移除')
            path=self.frame_path(identifier,frame_id)
            value['frames']=[f for f in value['frames'] if f['id']!=frame_id]
            value['calibration']=None;value['verification']=None;value['calibration_attempt']=None
            for frame in value['frames']: frame.pop('pose',None)
            self.recalculate(value);self.save(value);path.unlink(missing_ok=True)
            return self.public(value)

    def make_meal(self,identifier,options,paper_present,prepare_only=False,board_background=False):
        with self.lock:
            value=self.get(identifier)
            if value.get('meal_id'): return self.store.get(value['meal_id'])
            if value['status']!='complete': raise ValueError('请先结束扫描并检查照片')
            frames=[f for f in value['frames'] if f['observation']['quality']=='ok' and not f.get('duplicate_view')]
            minimum=int(self.jobs.settings.get('scan_minimum_frames',20))
            if len(frames)<minimum: raise ValueError(f'至少需要 {minimum} 张清晰且不重复的照片，当前 {len(frames)} 张')
            meal=self.store.create('scan',title=value['title'],scan_id=identifier,assistance=options,
                                   paper_present=paper_present,board_background=board_background)
            folder=self.store.root/'uploads'/meal['id'];source=folder/'capture';source.mkdir(parents=True)
            try:
                for i,frame in enumerate(frames): shutil.copyfile(self.frame_path(identifier,frame['id']),source/f'{i:06d}.jpg')
                manifest={'scan_id':identifier,'capture_frame_ids':[f['id'] for f in frames],
                          'calibration':value['calibration'],'coverage':value['coverage']}
                (source/'capture.json').write_text(json.dumps(manifest,ensure_ascii=False),encoding='utf-8')
                self.store.fields(meal['id'],input_file=str(source),source_photo=str(source/'000000.jpg'))
                self.jobs.submit(meal['id'],'scan-prepare' if prepare_only else 'scan',source)
                value['meal_id']=meal['id'];self.save(value)
            except Exception:
                self.jobs.cancel(meal['id']);self.store.delete(meal['id']);shutil.rmtree(folder);raise
            return self.store.get(meal['id'])

    def verify_scene(self,identifier,scene,source):
        """Only COLMAP-registered photos may support the final coverage map.

        Board poses are re-estimated with COLMAP's undistorted intrinsics;
        unregistered photos cannot retain an online green cell.
        """
        from food3d.io import read_json
        from food3d.schemas import load_scene
        _,cameras=load_scene(scene/'scene.json')
        manifest=read_json(source/'capture.json');mapping=manifest['capture_frame_ids']
        imported=read_json(scene/'frame_manifest.json')['frames']
        originals={f['frame_id']:int(Path(f['source_filename']).stem) for f in imported}
        verified=[]
        for camera in cameras:
            index=originals.get(camera.frame_id)
            if index is None or index>=len(mapping): continue
            with Image.open(scene/camera.image_path) as picture:
                observation=geometry.image_observation(picture.convert('RGB'))
            calibration={'K':camera.K,'distortion':[0]*5,'width':camera.width,'height':camera.height}
            verified.append({'id':mapping[index],'observation':observation,
                             'pose':geometry.estimate_pose(observation,calibration)})
        with self.lock:
            value=self.get(identifier)
            final=geometry.coverage(verified,None,value.get('origin_azimuth'))
            final['geometry_verified']=True;final['basis']='colmap_registered_photos_and_board_pnp'
            value['verification']={'registered_frames':len(cameras),'submitted_frames':len(mapping),
                'covered':final['covered'],'total':24,'coverage':final,'checked_at':now()}
            self.save(value)

    def purge(self):
        with self.lock:
            with self.store.connection() as db: db.execute('DELETE FROM scans')
            for directory in self.root.iterdir():
                if directory.is_dir() and re.fullmatch(r'[a-f0-9]{32}',directory.name): shutil.rmtree(directory)

    def remove_for_meal(self,meal_id):
        with self.lock:
            identifiers=[v['id'] for v in self.list() if v.get('meal_id')==meal_id]
            for identifier in identifiers:
                with self.store.connection() as db: db.execute('DELETE FROM scans WHERE id=?',(identifier,))
                folder=self.root/identifier
                if folder.is_dir(): shutil.rmtree(folder)
