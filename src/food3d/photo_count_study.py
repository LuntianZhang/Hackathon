"""Independent 5/10/30/60/90 RGB-only experiments, including failed small sets.

Each set owns its images, image-model answers, masks, intrinsics and camera poses.
All surviving cameras contribute to carving; no fixed 24-view cap is applied.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
from html import escape
import json
import math
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image

from .io import read_json,write_json
from .photos_only_volume import (Photo,import_photos,observe_photos,recognize_photos,
    fresh_segmentation,no_pose_experiment,camera_from_board,evenly_spaced)
from .schemas import CameraFrame,UNKNOWN_ID,save_scene
from .segmentation.vision_llm import validate_objects


def calibrate_subset(observations):
    """Same pixel/geometry gates for all budgets; explicitly allow a 3-view fit.

    This is experimental small-sample calibration, not a relaxation of the app's
    eight-view minimum. Small successful fits are labelled low-evidence below.
    """
    training=[o for i,o in enumerate(observations) if i%5!=0]
    heldout=[o['frame_id'] for i,o in enumerate(observations) if i%5==0]
    selected=[];descriptors=[]
    for observation in training:
        if len(observation['marker_ids'])<3:continue
        obj=np.asarray(observation['objects'],np.float32);pix=np.asarray(observation['pixels'],np.float32)
        h,_=cv2.findHomography(obj[:,:2],pix,cv2.RANSAC,2.5)
        if h is None:continue
        descriptor=cv2.perspectiveTransform(np.array([[[0,0],[.1,0],[0,.1]]],np.float32),h)[0].flatten()
        if any(np.linalg.norm(descriptor-d)<35 for d in descriptors):continue
        selected.append(observation);descriptors.append(descriptor)
    if len(selected)<3:raise ValueError(f'只有 {len(selected)} 张独立定位板标定视角，少于实验最低 3 张')
    width,height=selected[0]['width'],selected[0]['height']
    flags=(cv2.CALIB_FIX_ASPECT_RATIO|cv2.CALIB_ZERO_TANGENT_DIST|
           cv2.CALIB_FIX_K1|cv2.CALIB_FIX_K2|cv2.CALIB_FIX_K3)
    removed=[]
    while True:
        error,k,dist,rotations,translations=cv2.calibrateCamera(
            [np.asarray(o['objects'],np.float32) for o in selected],
            [np.asarray(o['pixels'],np.float32) for o in selected],(width,height),None,None,flags=flags)
        if error<=4 or len(selected)<=3:break
        residuals=[]
        for o,r,t in zip(selected,rotations,translations):
            pixels=cv2.projectPoints(np.asarray(o['objects'],np.float32),r,t,k,dist)[0].reshape(-1,2)
            residuals.append(float(np.sqrt(np.mean(np.sum((pixels-np.asarray(o['pixels']))**2,axis=1)))))
        removed.append(selected.pop(int(np.argmax(residuals)))['frame_id'])
    normals=[cv2.Rodrigues(r)[0][:,2] for r in rotations]
    diversity=max(math.degrees(math.acos(float(np.clip(a@b,-1,1)))) for a in normals for b in normals)
    if not np.isfinite(k).all() or not np.isfinite(error) or error>4:raise ValueError('本组独立内参标定重投影误差超过 4 像素')
    if diversity<12:raise ValueError('本组标定视角倾斜差异不足 12 度')
    if not .35*width<k[0,0]<4*width:raise ValueError('本组独立估计焦距不合理')
    if not .35*width<k[0,2]<.65*width or not .35*height<k[1,2]<.65*height:
        raise ValueError('本组独立估计主点偏离图像中心过多')
    return {'K':k.tolist(),'distortion':dist.flatten().tolist(),'width':width,'height':height,
        'reprojection_error':float(error),'views':len(selected),'tilt_range_deg':diversity,
        'training_frame_ids':[o['frame_id'] for o in selected],'rejected_training_frames':removed,
        'heldout_frame_ids':heldout,'old_intrinsics_or_poses_used':False,
        'small_sample_fit':len(selected)<8,'intrinsics_maximum_views':None}


def canonical_groups(evidence):
    mouse=[o['instance_id'] for o in evidence['objects'] if o['kind']=='non_food' and
           ('mouse' in o['label'].lower() or '鼠标' in o['name'])]
    packages=[o['instance_id'] for o in evidence['objects'] if o['kind']=='packaged_food']
    if not mouse or not packages:raise ValueError('本组未同时识别到鼠标和包装，不能做相同对象范围的比较')
    return {1:mouse,2:packages}


def grouped_mask(labels,groups):
    result=np.zeros_like(labels)
    result[labels==UNKNOWN_ID]=UNKNOWN_ID
    selected=[]
    for identifier,source in groups.items():
        result[np.isin(labels,source)]=identifier;selected.extend(source)
    result[(labels!=0)&(~np.isin(labels,[*selected,UNKNOWN_ID]))]=UNKNOWN_ID
    return result


def merge_group_anchors(anchors,groups):
    """Merge package identities BEFORE ambiguity detection, not after it."""
    merged={}
    for fid,objects in anchors.items():
        for identifier,source in groups.items():
            masks=[objects[i] for i in source if i in objects]
            if masks:merged.setdefault(fid,{})[identifier]=np.logical_or.reduce(masks)
    return merged


def track_measurement_groups(photos,evidence,output,checkpoint):
    from .segmentation.llm_segment import track_anchors
    anchors={}
    for fid in evidence['keyframes']:
        for obj in evidence['objects']:
            path=output/'segmentation/anchors'/f"{fid}_{obj['instance_id']}.png"
            if path.is_file():anchors.setdefault(fid,{})[obj['instance_id']]=np.asarray(Image.open(path))>0
    groups=canonical_groups(evidence);anchors=merge_group_anchors(anchors,groups)
    folder=output/'measurement_segmentation';folder.mkdir()
    for name in ('frames','masks','confidence','anchors'):(folder/name).mkdir()
    for index in range(len(photos)):
        shutil.copyfile(output/'segmentation/frames'/f'{index:06d}.jpg',folder/'frames'/f'{index:06d}.jpg')
    for fid,objects in anchors.items():
        for identifier,mask in objects.items():Image.fromarray(mask.astype(np.uint8)*255).save(folder/'anchors'/f'{fid}_{identifier}.png')
    counts=track_anchors(photos,anchors,folder,checkpoint,[1,2])
    write_json(output/'measurement_segmentation.json',{'groups':groups,'frames':len(photos),
        'method':'union source SAM anchors before tracking and before inter-object ambiguity detection',
        'fresh_within_this_photo_budget':True,'counts':counts})
    return folder


def geometry_case(photos,observations,evidence,output,resolution=224,grouped_segmentation=None):
    from .silhouette_volume import estimate_visual_hull
    report={'requested_photos':len(photos),'resolution':resolution,'status':'failed_quality_gate',
        'old_intrinsics_poses_masks_geometry_used':False,'used_views':0,'pose_diagnostics':[],
        'excluded_frames':[],'instances':[],'carving_view_cap':None,
        'pose_limits_px':{'fit_rms':4,'heldout_marker_rms':8}}
    cv2.setRNGSeed(1700+len(photos))
    try:
        calibration=calibrate_subset(observations);report['calibration']=calibration
        groups=canonical_groups(evidence);report['semantic_groups']=groups
        report['measurement_grouping']='merged SAM anchors before tracking' if grouped_segmentation else 'post-hoc label union'
        frames=[];masks=[];folder=output/'geometry';folder.mkdir(exist_ok=True)
        (folder/'masks').mkdir(exist_ok=True)
        for photo,observation in zip(photos,observations):
            try:
                pose,diagnostic=camera_from_board(observation,calibration)
                _,held=camera_from_board(observation,calibration,holdout=True)
                diagnostic['heldout_marker_rms_px']=held['heldout_marker_rms_px']
                if diagnostic['heldout_marker_rms_px']>8:raise ValueError('留出标记误差超过 8 像素')
                if grouped_segmentation is None:
                    labels=grouped_mask(np.asarray(Image.open(output/'segmentation/masks'/f'{photo.frame_id}.png')),groups)
                else:labels=np.asarray(Image.open(grouped_segmentation/'masks'/f'{photo.frame_id}.png'))
                ys,xs=np.nonzero(np.isin(labels,[1,2]))
                if len(xs)<16:raise ValueError('没有有效前景')
                if xs.min()<2 or ys.min()<2 or xs.max()>=photo.width-2 or ys.max()>=photo.height-2:
                    raise ValueError('前景被照片边缘裁切')
                frame=CameraFrame(photo.frame_id,'../input/'+photo.path.name,photo.width,photo.height,
                    calibration['K'],pose.tolist())
                frames.append(frame);masks.append(labels);report['pose_diagnostics'].append(diagnostic)
                Image.fromarray(labels).save(folder/'masks'/f'{photo.frame_id}.png')
            except (ValueError,cv2.error) as error:
                report['excluded_frames'].append({'frame_id':photo.frame_id,'reason':str(error)})
        report['used_views']=len(frames);report['used_frame_ids']=[f.frame_id for f in frames]
        if len(frames)<3:raise ValueError(f'只有 {len(frames)} 张照片通过质量筛选，无法进行多视图体积计算')
        save_scene(folder/'scene.json',frames,source='photo_count_study_fresh_board',world_units='metres')
        result=estimate_visual_hull(frames,masks,[1,2],resolution=resolution,
            plane={'point':[0,0,0],'normal':[0,0,1]})
        for instance in result['instances']:
            instance['volume_ml']=instance['volume_occlusion_aware_units3']*1e6
            instance['name']='鼠标' if instance['instance_id']==1 else '两包零食合并外轮廓'
        report.update(status='estimated_low_evidence' if len(frames)<8 or calibration['small_sample_fit'] else 'estimated',
            instances=result['instances'],carving=result)
        if any(o['status']!='estimated' for o in result['instances']):
            report['status']='failed_volume_quality_gate';report['reason']='轮廓体积为空或搜索边界被截断'
    except (ValueError,cv2.error) as error:report['reason']=str(error)
    write_json(output/'fresh_pose.json',report)
    return report


def case_recognition(case,resume):
    output=case['output'];photos=case['photos'];obs=case['observations']
    if resume and (output/'agent/recognition_answer.json').is_file():
        selected=evenly_spaced([p for p,o in zip(photos,obs) if len(o['marker_ids'])>=3],6)
        views=[(p.frame_id,Image.open(p.path).convert('RGB')) for p in selected]
        raw=read_json(output/'agent/recognition_answer.json')
        evidence={'objects':validate_objects(raw,[f for f,_ in views]),'model':'current-agent',
            'keyframes':[f for f,_ in views],'provider':'codex','old_results_used':False}
        return evidence,views
    return recognize_photos(photos,obs,output)


def write_reports(output,results,audit):
    ordered=sorted(results,key=lambda r:r['input_photos'])
    report={'study':'independent_input_photo_count','counts':[r['input_photos'] for r in ordered],
        'sampling':'independently uniform over raw filename chronology, before any pose fitting',
        'no_shared_intrinsics_poses_masks_or_image_model_answers':True,
        'all_accepted_cameras_used_for_carving':True,'baseline_used':False,
        'package_union_before_tracking':True,
        'parameter_control':'same 224 grid, 95% mask agreement, 1px dilation, min 3 positive views',
        'audit':audit,'results':ordered,'no_measured_physical_volume_ground_truth':True}
    write_json(output/'study.json',report)
    rows=[];rough=[]
    for case in ordered:
        g=case.get('fresh_pose',{});values={i['instance_id']:i for i in g.get('instances',[])}
        valid=g.get('status') in ('estimated','estimated_low_evidence')
        vols=[f"{values[i]['volume_ml']:.1f}" if valid and i in values else '未通过' for i in (1,2)]
        state='低样本粗估' if g.get('status')=='estimated_low_evidence' else '通过' if valid else escape(g.get('reason',case.get('error','失败')))
        rows.append(f'<tr><td>{case["input_photos"]}</td><td>{case.get("recognition_views",0)}</td><td>{g.get("calibration",{}).get("views",0)}</td><td>{g.get("used_views",0)}</td><td>{vols[0]}</td><td>{vols[1]}</td><td>{state}</td></tr>')
        items=case.get('no_pose',{}).get('instances',[])
        text=[]
        for item in items:
            number=item.get('volume_ml');interval=item.get('prior_sensitivity_ml')
            value='无法可靠估算' if number is None else f'{number:.1f} mL（先验 {interval[0]:.1f}–{interval[1]:.1f}）'
            text.append(escape(item['name'])+'：'+value)
        rough.append(f'<tr><td>{case["input_photos"]}</td><td>{"<br>".join(text) or escape(case.get("no_pose",{}).get("reason","失败"))}</td></tr>')
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>不同照片数量体积实验</title>
<style>body{{font:16px/1.7 "Microsoft YaHei",sans-serif;background:#edf2ed;color:#183d3b;margin:0}}main{{max-width:1180px;margin:32px auto;background:white;padding:28px;border-radius:20px}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:10px;border-bottom:1px solid #d7e2dc;text-align:left}}th{{background:#edf4ef}}.note{{background:#fff3d6;padding:18px;border-radius:12px}}a{{color:#17615a}}</style>
<main><h1>5、10、30、60、90 张照片的独立体积实验</h1>
<p>每组均从原始照片按时间均匀抽样，先抽样再做相机估计。每组单独识别、分割和标定；没有跨组借用焦距、位姿、掩码或模型答案。定位板按 20 × 20 cm、标记按 3 cm 计算。结果独立于 App。</p>
<p class="note">本次让全部通过质量校验的照片参与轮廓体积，不再限制为 24 个视角。表中“输入照片数”和“实际计算视角数”需要分开看：增加输入不保证全部能用于几何计算。mL = cm³。</p>
<table><tr><th>输入照片</th><th>模型识别关键帧</th><th>内参拟合帧</th><th>实际体积视角</th><th>鼠标 mL</th><th>两包合并外轮廓 mL</th><th>状态</th></tr>{''.join(rows)}</table>
<p>所有组固定网格 224、轮廓一致性 95%、掩码扩张 1 像素，至少 3 个正向轮廓支持；相机拟合误差限 4 像素、留出标记误差限 8 像素。5 张组允许尝试最低 3 张的实验标定，保持同样的误差与角度门槛，并标记少样本结果；没有改动 App 的标定门槛。</p>
<p>每组最多 6 张关键帧进入大模型联合识别，其余照片由本组 SAM2 传播得到掩码。几何测量先合并两包的 SAM 掩码，再作为同一个对象传播，避免两包相互重叠时被标成归属不确定；测量的是合并外轮廓，不是包内食物净体积。</p>
<h2>完全无位姿：形状先验粗估</h2><p>厚度与填充比例仍是视觉模型先验，不是照片直接测得的三维高度。范围只表示形状假设变化，不能解释成置信区间。参考帧掩码过小或归属不确定比例过高时拒绝给出体积；通过此检查也不证明隐藏轮廓完整。遮挡严重的下层包装可能无法估计，各组对象划分也可能不同。</p>
<table><tr><th>输入照片</th><th>各对象粗估</th></tr>{''.join(rough)}</table>
<p class="note">没有实测物理体积真值。照片数增加后数值接近只能说明相对稳定，不能证明准确。单次独立运行还混合了抽样、识别、分割和标定变化；这不是只改变轮廓视角数的受控实验。</p>
<p><a href="study.json">完整实验数据</a> · <a href="study_manifest.json">抽样配置与输入来源</a></p></main></html>'''
    (output/'report.html').write_text(html,encoding='utf-8')
    return report


def run(input_dir,output,checkpoint,counts=(5,10,30,60,90),resume=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    cases=[]
    for count in counts:
        folder=output/f'n{count:03d}';folder.mkdir(exist_ok=True)
        if resume and (folder/'input_manifest.json').is_file():
            manifest=read_json(folder/'input_manifest.json');photos=[]
            for p in manifest['photos']:
                path=(folder/p['input_file']).resolve()
                if not path.is_relative_to(folder/'input'):raise ValueError('恢复输入不在本组隔离目录内')
                if hashlib.sha256(path.read_bytes()).hexdigest()!=p['sha256']:raise ValueError('恢复输入校验和不匹配')
                photos.append(Photo(p['frame_id'],path,p['width'],p['height']))
            if len(photos)!=count:raise ValueError('恢复输入数量不匹配')
        else:photos=import_photos(input_dir,folder,count=count)
        observations=observe_photos(photos);write_json(folder/'board_observations.json',observations)
        cases.append({'count':count,'photos':photos,'observations':observations,'output':folder})
    write_json(output/'study_manifest.json',{'counts':list(counts),'input_directory':str(Path(input_dir).resolve()),
        'sampling':'uniform raw filename chronology before pose filtering','resolution':224,'agreement':.95,
        'dilation_pixels':1,'minimum_positive_views':3,'shared_previous_scene_data':False})
    # After RGB import, deny reads of all other uploaded data and old experiments.
    data=Path(__file__).resolve().parents[2]/'data';blocked=[data/'pansense',data/'outputs']
    audit={'previous_scene_read_attempts':0,'previous_scene_reads':0,'deny_previous_data_enabled':True}
    def hook(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes)):return
        mode=args[1];flags=args[2]
        reading=isinstance(mode,str) and 'r' in mode or mode is None and isinstance(flags,int) and flags&3==0
        if not reading:return
        path=Path(args[0]).resolve()
        if not path.is_relative_to(output) and any(path.is_relative_to(p) for p in blocked):
            audit['previous_scene_read_attempts']+=1
            raise PermissionError('本实验禁止读取其他餐盘或旧实验文件')
    sys.addaudithook(hook)
    results=[];by_count={};no_pose_futures={}
    started=time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as inference:
        recognition_futures={inference.submit(case_recognition,case,resume):case for case in cases}
        for future in as_completed(recognition_futures):
            case=recognition_futures[future];count=case['count'];folder=case['output'];mark=time.monotonic()
            result={'input_photos':count,'source_names':[p['source_name'] for p in read_json(folder/'input_manifest.json')['photos']]}
            by_count[count]=result;results.append(result)
            try:
                recognition=future.result();result['recognition_views']=len(recognition[1])
                print(f'n={count}: recognition complete, {len(recognition[0]["objects"])} objects',flush=True)
                if resume and (folder/'segmentation_summary.json').is_file():
                    evidence=read_json(folder/'recognition.json');views=recognition[1]
                else:
                    evidence,views=fresh_segmentation(case['photos'],case['observations'],folder,checkpoint,recognition)
                result['object_names']=[o['name'] for o in evidence['objects']]
                def prior_task(case=case,evidence=evidence,views=views):
                    try:
                        prior_path=case['output']/'agent/shape_prior_answer.json'
                        prior=read_json(prior_path) if resume and prior_path.is_file() else None
                        return no_pose_experiment(case['photos'],case['observations'],evidence,views,case['output'],prior=prior)
                    except (ValueError,RuntimeError) as error:return {'status':'failed','reason':str(error),'instances':[]}
                no_pose_futures[inference.submit(prior_task)]=case
                if resume and (folder/'measurement_segmentation.json').is_file():
                    measurement=folder/'measurement_segmentation'
                else:measurement=track_measurement_groups(case['photos'],evidence,folder,checkpoint)
                result['fresh_pose']=geometry_case(case['photos'],case['observations'],evidence,folder,
                    grouped_segmentation=measurement)
                print(f'n={count}: geometry {result["fresh_pose"]["status"]}, used {result["fresh_pose"]["used_views"]} views, '+
                    ', '.join(f"{o['name']} {o['volume_ml']:.1f} mL" for o in result['fresh_pose']['instances']),flush=True)
            except Exception as error:
                result['error']=str(error);print(f'n={count}: FAILED {error}',flush=True)
            result['segmentation_geometry_wall_seconds']=time.monotonic()-mark
            write_json(folder/'case_report.json',result)
        for future in as_completed(no_pose_futures):
            case=no_pose_futures[future];result=by_count[case['count']]
            try:result['no_pose']=future.result()
            except Exception as error:result['no_pose']={'status':'failed','reason':str(error),'instances':[]}
            write_json(case['output']/'case_report.json',result)
            print(f'n={case["count"]}: no-pose experiment complete',flush=True)
    audit['total_wall_seconds']=time.monotonic()-started
    report=write_reports(output,results,audit)
    print('STUDY COMPLETE '+str(output/'report.html'),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=Path('.tools/weights/sam2.1_hiera_small.pt'))
    parser.add_argument('--counts',type=int,nargs='+',default=[5,10,30,60,90]);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();run(args.input,args.output,args.checkpoint,args.counts,args.resume)
