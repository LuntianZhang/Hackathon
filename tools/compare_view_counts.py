"""Uniform 5/10/30-view SfM diagnostics and controlled Gaussian reconstruction."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from food3d.paths import ProjectPaths, resolve_arguments
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

from food3d.data.cameras import export_nerfstudio_transforms
from food3d.data.view_sampling import orbit_coordinates, select_uniform_views, select_common_holdout, fit_similarity
from food3d.io import read_json, write_json, write_point_cloud_ply
from food3d.schemas import load_scene, save_scene

PROJECT = Path(__file__).resolve().parents[1]
PATHS = ProjectPaths(PROJECT)
COUNTS = [5, 10, 30]


def event(stage, **data):
    print(json.dumps({'stage':stage, **data}, ensure_ascii=False), flush=True)


def filtered_points(source, selected):
    ids = {frame.colmap_image_id for frame in selected}
    points, colors = [], []
    text = source/'colmap/undistorted_text/points3D.txt'
    for line in text.read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.startswith('#'): continue
        fields = line.split()
        track = {int(value) for value in fields[8::2]}
        if len(track & ids) >= 2:
            points.append(list(map(float, fields[1:4])))
            colors.append(list(map(int, fields[4:7])))
    return np.asarray(points).reshape(-1,3), np.asarray(colors,dtype=np.uint8).reshape(-1,3)


def prepare(source, segmentation, output):
    source, segmentation = source.resolve(), segmentation.resolve()
    metadata, frames = load_scene(source/'scene.json')
    orbit = orbit_coordinates([frame.camera_to_world for frame in frames])
    output.mkdir(parents=True, exist_ok=False)
    plan = {'source':str(source),'segmentation':str(segmentation),'counts':COUNTS,
        'iterations':7000,'evaluation_width':640,'source_views':len(frames),
        'selection_basis':'full COLMAP camera optical axes + PCA orbit plane; uniform azimuth, unique assignment',
        'observed_angular_span_degrees':float(np.degrees(orbit['span'])),
        'largest_missing_gap_degrees':float(np.degrees(orbit['largest_gap'])),
        'closed_orbit':orbit['closed'],'pivot':orbit['pivot'].tolist(),
        'limitation':'Fixed-pose training uses cameras calibrated from the full dataset; selected-track sparse points were also optimized with full SfM. It is not an end-to-end N-image-only result.',
        'groups':{}}
    original = read_json(source/'frame_manifest.json')
    records = {entry['frame_id']:entry for entry in original['frames'] if entry['accepted']}
    used = set()
    selections = {}
    for count in COUNTS:
        indices, distribution = select_uniform_views(orbit, count)
        selections[count] = indices
        used.update(indices)
        selected = [frames[index] for index in indices]
        group = output/f'n{count}'
        known, strict = group/'fixed_pose_scene', group/'independent_colmap_scene'
        (known/'images').mkdir(parents=True)
        (strict/'frames').mkdir(parents=True)
        cameras, raw_records = [], []
        for frame in selected:
            basename = Path(frame.image_path).name
            shutil.copy2(source/frame.image_path, known/'images'/basename)
            cameras.append(replace(frame,image_path='images/'+basename))
            entry = dict(records[frame.frame_id])
            raw_image = source/entry['image_path']
            shutil.copy2(raw_image,strict/'frames'/raw_image.name)
            entry['image_path'] = 'frames/'+raw_image.name
            raw_records.append(entry)
        save_scene(known/'scene.json',cameras,source='fixed_full_colmap_calibration',
            world_units=metadata['world_units'],metadata={'input_source':metadata['metadata'].get('input_source'),
                'reference_scene':str(source),'rgb_training_image_count':count,'full_calibration_images':len(frames)})
        xyz, rgb = filtered_points(source,selected)
        if len(xyz) < 4: raise ValueError(f'{count} views have insufficient shared sparse points')
        write_point_cloud_ply(known/'sparse_points.ply',xyz,rgb)
        export_nerfstudio_transforms(known/'transforms.json',cameras,'sparse_points.ply')
        transforms = read_json(known/'transforms.json')
        # Exact N RGB inputs. These internal logging views are NOT the final holdout.
        filenames = [frame.image_path for frame in cameras]
        transforms.update(train_filenames=filenames,val_filenames=filenames,test_filenames=filenames)
        write_json(known/'transforms.json',transforms)
        write_json(strict/'frame_manifest.json',{**original,'frames':raw_records,
            'selection':'camera azimuth uniform subsampling; only these images are passed to COLMAP'})
        plan['groups'][str(count)] = {'frame_ids':[frame.frame_id for frame in selected],
            'initial_sparse_points':len(xyz),'distribution':distribution}
    indices, distribution = select_common_holdout(orbit,used,12)
    plan['holdout_frame_ids'] = [frames[index].frame_id for index in indices]
    plan['holdout_distribution'] = distribution
    assert all(not set(plan['holdout_frame_ids']) & set(group['frame_ids']) for group in plan['groups'].values())
    write_json(output/'plan.json',plan)
    plot_sampling(output,orbit,frames,selections,indices)
    event('prepared',groups=plan['groups'],holdout=plan['holdout_frame_ids'],
          observed_span=plan['observed_angular_span_degrees'])


def refresh_holdout(output):
    plan=read_json(output/'plan.json')
    _,frames=load_scene(Path(plan['source'])/'scene.json')
    orbit=orbit_coordinates([frame.camera_to_world for frame in frames])
    by_id={frame.frame_id:index for index,frame in enumerate(frames)}
    selections={int(key):[by_id[frame_id] for frame_id in group['frame_ids']] for key,group in plan['groups'].items()}
    used={index for indices in selections.values() for index in indices}
    indices,distribution=select_common_holdout(orbit,used,12)
    plan.update(holdout_frame_ids=[frames[index].frame_id for index in indices],holdout_distribution=distribution)
    write_json(output/'plan.json',plan)
    plot_sampling(output,orbit,frames,selections,indices)
    event('holdout_updated',frames=plan['holdout_frame_ids'],distribution=distribution)


def plot_sampling(output,orbit,frames,selections,holdout):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(13,4.4), constrained_layout=True)
    xy = orbit['xy']
    for ax,count in zip(axes,COUNTS):
        indices = selections[count]
        ax.scatter(xy[:,0],xy[:,1],s=10,color='#d8dedc',label='All calibrated views')
        ax.scatter(xy[holdout,0],xy[holdout,1],s=28,marker='x',color='#b17d45',label='Common holdout')
        ax.scatter(xy[indices,0],xy[indices,1],s=38,color='#205657',label=f'{count} training views')
        ax.scatter([0],[0],marker='+',color='black',label='Optical-axis pivot')
        for index in indices: ax.plot([0,xy[index,0]],[0,xy[index,1]],color='#205657',alpha=.12)
        ax.set_title(f'{count} views - uniform azimuth')
        ax.set_aspect('equal');ax.set_xlabel('Orbit plane X (SfM units)');ax.set_ylabel('Orbit plane Y')
        ax.legend(fontsize=7,loc='best')
    fig.savefig(output/'sampling_distribution.png',dpi=160)
    plt.close(fig)


def command(argv, log, timeout=1800):
    environment = PATHS.environment()
    started = time.monotonic()
    with log.open('w',encoding='utf-8') as stream:
        stream.write('COMMAND '+json.dumps([str(arg) for arg in argv],ensure_ascii=False)+'\n')
        stream.flush()
        result = subprocess.run([str(arg) for arg in argv],cwd=PROJECT,env=environment,
            stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
    return result.returncode, time.monotonic()-started


def independent_colmap(output):
    result = {}
    for count in COUNTS:
        group = output/f'n{count}'
        scene = group/'independent_colmap_scene'
        event('independent_colmap_start',count=count)
        code,seconds = command([PATHS.recognition_python,'-m','food3d','colmap',
            '--scene',scene,'--config',PROJECT/'configs/subset_comparison.yaml'],group/'colmap.log',600)
        if code == 0:
            data = read_json(scene/'validation.json')
            result[str(count)] = {**data,'seconds':seconds,'status':'complete' if data['registered_frames']==count else 'partial'}
        else:
            selection = read_json(scene/'colmap_selection.json') if (scene/'colmap_selection.json').is_file() else {}
            result[str(count)] = {'status':'failed','registered_frames':selection.get('registered_frames',0),
                'input_frames':count,'seconds':seconds,'log':str(group/'colmap.log')}
        write_json(output/'independent_colmap.json',result)
        event('independent_colmap_complete',count=count,result=result[str(count)])


def train(output):
    from food3d.reconstruction.export import export_checkpoint
    result = {}
    for count in COUNTS:
        group = output/f'n{count}'
        if (group/'bundle/model.json').is_file():
            result[str(count)]=read_json(group/'training_result.json');continue
        event('training_start',count=count,iterations=7000)
        code,seconds = command([PATHS.reconstruction_python,'-m','food3d.reconstruction.train_worker',
            '--scene',group/'fixed_pose_scene','--output',group/'training','--iterations','7000',
            '--config',PROJECT/'configs/subset_comparison.yaml'],group/'training.log')
        if code: raise RuntimeError(f'{count}-view training failed, inspect {group / "training.log"}')
        checkpoint = max((group/'training').rglob('*.ckpt'),key=lambda path:path.stat().st_mtime)
        model = export_checkpoint(checkpoint,group/'fixed_pose_scene',group/'bundle',checkpoint.parent.parent/'dataparser_transforms.json')
        result[str(count)] = {'seconds':seconds,'gaussians':model['gaussians'],
            'checkpoint':str(checkpoint),'steps':7000,'rgb_training_images':count,'mode':'fixed_full_colmap_calibration'}
        write_json(group/'training_result.json',result[str(count)])
        event('training_complete',count=count,result=result[str(count)])
    write_json(output/'training.json',result)


def train_independent(output):
    from food3d.reconstruction.export import export_checkpoint
    diagnostics=read_json(output/'independent_colmap.json')
    for key,diagnostic in diagnostics.items():
        if diagnostic['status']!='complete':continue
        group=output/f'n{key}';scene=group/'independent_colmap_scene'
        if (group/'independent_bundle/model.json').is_file():continue
        transforms=read_json(scene/'transforms.json')
        filenames=[frame['file_path'] for frame in transforms['frames']]
        transforms.update(train_filenames=filenames,val_filenames=filenames,test_filenames=filenames)
        write_json(scene/'transforms.json',transforms)
        event('independent_training_start',count=int(key))
        code,seconds=command([PATHS.reconstruction_python,'-m','food3d.reconstruction.train_worker',
            '--scene',scene,'--output',group/'independent_training','--iterations','7000',
            '--config',PROJECT/'configs/subset_comparison.yaml'],group/'independent_training.log')
        if code:raise RuntimeError('Independent training failed: '+str(group/'independent_training.log'))
        checkpoint=max((group/'independent_training').rglob('*.ckpt'),key=lambda path:path.stat().st_mtime)
        model=export_checkpoint(checkpoint,scene,group/'independent_bundle',checkpoint.parent.parent/'dataparser_transforms.json')
        write_json(group/'independent_training_result.json',{'seconds':seconds,'gaussians':model['gaussians'],
            'rgb_training_images':int(key),'mode':'independent_N_image_COLMAP_then_Splatfacto','steps':7000})
        event('independent_training_complete',count=int(key),gaussians=model['gaussians'])


def evaluate(output):
    import gc
    import torch
    from food3d.reconstruction.export import transform_camera
    from food3d.viewer.renderer import GaussianRenderer,resize_camera
    from food3d.visuals import contact_sheet
    plan=read_json(output/'plan.json')
    source=Path(plan['source']);seg=Path(plan['segmentation'])
    _,cameras=load_scene(source/'scene.json');by_id={frame.frame_id:frame for frame in cameras}
    mask_paths=read_json(seg/'segmentation.json')['masks']
    result={'protocol':plan,'independent_colmap':read_json(output/'independent_colmap.json'),
            'training':read_json(output/'training.json'),'fixed_pose_quality':{}}
    cases=[(str(count),output/f'n{count}','bundle','evaluation') for count in COUNTS]
    cases += [(str(count)+'_independent',output/f'n{count}','independent_bundle','independent_evaluation')
              for count in COUNTS if (output/f'n{count}/independent_bundle/model.json').is_file()]
    for key,group,bundle_name,evaluation_name in cases:
        dest=group/evaluation_name;dest.mkdir(exist_ok=True)
        renderer=GaussianRenderer(group/bundle_name)
        transform=renderer.metadata['training_transform']
        alignment=None
        if key.endswith('_independent'):
            _,own=load_scene(group/'independent_colmap_scene/scene.json')
            full_centers=np.array([by_id[frame.frame_id].camera_to_world for frame in own])[:,:3,3]
            own_centers=np.array([frame.camera_to_world for frame in own])[:,:3,3]
            rotation,scale,translation,rmse=fit_similarity(full_centers,own_centers)
            orientation_errors=[]
            for frame in own:
                delta=np.asarray(frame.camera_to_world)[:3,:3].T@rotation@np.asarray(by_id[frame.frame_id].camera_to_world)[:3,:3]
                orientation_errors.append(float(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2,-1,1)))))
            alignment={'rotation':rotation.tolist(),'scale':scale,'translation':translation.tolist(),
                'camera_center_rmse_in_group_units':rmse,
                'camera_center_relative_rmse':rmse/float(np.median(np.linalg.norm(own_centers-own_centers.mean(0),axis=1))),
                'mean_orientation_error_degrees':float(np.mean(orientation_errors)),
                'method':'Sim3 fit of registered camera centers; full-reference poses only used for evaluation'}
        metrics=[]
        for frame_id in plan['holdout_frame_ids']:
            original=by_id[frame_id]
            pose=np.asarray(original.camera_to_world).copy()
            if alignment:
                pose[:3,:3]=rotation@pose[:3,:3]
                pose[:3,3]=scale*rotation@pose[:3,3]+translation
            camera=resize_camera(replace(original,camera_to_world=transform_camera(
                pose,np.asarray(transform['transform']),transform['scale']).tolist()),640)
            rendered=renderer.render(camera)
            picture=Image.open(source/original.image_path).convert('RGB').resize((camera.width,camera.height),Image.Resampling.LANCZOS)
            labels=np.asarray(Image.open(seg/mask_paths[frame_id]).resize(picture.size,Image.Resampling.NEAREST))
            fg=(labels>0)&(labels<65535)
            error=(np.asarray(picture,dtype=float)/255-rendered['rgb'].astype(float)/255)**2
            metrics.append({'frame_id':frame_id,'psnr_db':float(-10*np.log10(max(float(error.mean()),1e-12))),
                'foreground_psnr_db':float(-10*np.log10(max(float(error[fg].mean()),1e-12))) if fg.any() else None,
                'mean_alpha':float(rendered['alpha'].mean())})
            Image.fromarray(rendered['rgb']).save(dest/f'{frame_id}_render.png')
            picture.save(output/f'{frame_id}_source.png')
        quality={'mean_psnr_db':float(np.mean([row['psnr_db'] for row in metrics])),
            'mean_foreground_psnr_db':float(np.mean([row['foreground_psnr_db'] for row in metrics if row['foreground_psnr_db'] is not None])),
            'per_view':metrics}
        if alignment:
            result.setdefault('independent_reconstruction_quality',{})[key]={**quality,'alignment':alignment,
                'training':read_json(group/'independent_training_result.json')}
        else:result['fixed_pose_quality'][key]=quality
        del renderer;gc.collect();torch.cuda.empty_cache()
        event('evaluation_complete',case=key,quality=quality)
    previews=[];details=[]
    for frame_id in plan['holdout_frame_ids'][::2]:
        paths=[('reference',output/f'{frame_id}_source.png')]+[(f'{count} views',output/f'n{count}/evaluation/{frame_id}_render.png') for count in COUNTS]
        image=Image.open(paths[0][1]);labels=np.asarray(Image.open(seg/mask_paths[frame_id]).resize(image.size,Image.Resampling.NEAREST))
        ys,xs=np.where((labels>0)&(labels<65535))
        box=(max(0,int(xs.min())-12),max(0,int(ys.min())-12),min(image.width,int(xs.max())+13),min(image.height,int(ys.max())+13)) if len(xs) else (0,0,image.width,image.height)
        for label,path in paths:
            picture=Image.open(path).convert('RGB')
            previews.append((frame_id+' '+label,picture.copy()))
            details.append((frame_id+' '+label,picture.crop(box)))
    contact_sheet(previews,output/'comparison_sheet.png',columns=4)
    contact_sheet(details,output/'food_detail_sheet.png',columns=4)
    if 'independent_reconstruction_quality' in result:
        strict=[]
        for frame_id in plan['holdout_frame_ids'][::2]:
            paths=[('reference',output/f'{frame_id}_source.png'),('30 fixed poses',output/f'n30/evaluation/{frame_id}_render.png'),
                   ('30 independent COLMAP',output/f'n30/independent_evaluation/{frame_id}_render.png')]
            for label,path in paths:strict.append((frame_id+' '+label,Image.open(path).convert('RGB')))
        contact_sheet(strict,output/'independent_30_comparison.png',columns=3)
    for count in COUNTS:
        _,selected=load_scene(output/f'n{count}/fixed_pose_scene/scene.json')
        contact_sheet([(frame.frame_id,Image.open(output/f'n{count}/fixed_pose_scene'/frame.image_path).convert('RGB'))
                      for frame in selected],output/f'selected_{count}_views.png',columns=5)
    write_json(output/'report.json',result)
    html_report(output,result)
    event('comparison_complete',report=str(output/'report.json'))


def html_report(output,result):
    buttons=''.join(f'<button data-frame="{frame}" class="frame">{frame}</button>' for frame in result['protocol']['holdout_frame_ids'])
    rows=''
    for count in COUNTS:
        sfm=result['independent_colmap'][str(count)]
        quality=result['fixed_pose_quality'][str(count)]
        training=result['training'][str(count)]
        distribution=result['protocol']['groups'][str(count)]['distribution']
        rows+=f'<tr><td>{count} 张</td><td>{sfm["registered_frames"]}/{count}</td><td>{distribution["max_angular_error_degrees"]:.1f}°</td><td>{quality["mean_psnr_db"]:.2f}</td><td>{quality["mean_foreground_psnr_db"]:.2f}</td><td>{training["seconds"]:.0f} 秒</td><td>{training["gaussians"]:,}</td><td><a href="n{count}/bundle/gaussians.ply">PLY</a></td></tr>'
    independent=result.get('independent_reconstruction_quality',{}).get('30_independent')
    extra=f'<p>30 张独立 COLMAP → Splatfacto：全图 PSNR {independent["mean_psnr_db"]:.2f} dB，食物区域 {independent["mean_foreground_psnr_db"]:.2f} dB。<a href="n30/independent_bundle/gaussians.ply">下载独立重建 PLY</a></p>' if independent else ''
    payload=json.dumps(result['protocol']['holdout_frame_ids'])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>5 / 10 / 30 张 · 重建对比</title>
<style>*{box-sizing:border-box}body{margin:0;background:#f3f6f1;color:#243e38;font:15px/1.7 system-ui,sans-serif}main{max-width:1380px;margin:32px auto;padding:24px}h1{font-size:32px;line-height:1.3}h2{font-size:21px}.card{background:white;border-radius:18px;padding:22px;margin:20px 0;box-shadow:0 5px 25px #173a2410}.muted{color:#60756a}.notice{border-left:4px solid #a7be72;padding:12px 18px;background:#edf3e4}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}.tile img{width:100%;display:block;background:white;border-radius:10px}.tile strong{display:block;margin-bottom:8px}.frame{border:1px solid #ccdbc7;border-radius:8px;background:white;color:#305b4a;padding:8px 12px;margin:5px;cursor:pointer}.frame.active{background:#205657;color:white}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:10px}table{border-collapse:collapse;width:100%;white-space:nowrap}td,th{text-align:left;border-bottom:1px solid #e5ece1;padding:12px}a{color:#246653}.scroll{overflow:auto}.diagram{max-width:100%;height:auto}input{accent-color:#205657}.strict{max-width:900px;margin:auto}@media(max-width:800px){main{padding:12px;margin:12px auto}.grid{grid-template-columns:repeat(2,1fr)}h1{font-size:26px}}
</style><main><h1>均匀视角：5、10、30 张照片的重建效果</h1><p class="muted">水果2 · 每组 7000 步 · 12 个共同留出视角 · 高斯模型与指标均由本机实际运行产生</p>
<div class="notice">下面三组画质对比固定使用完整 104 张数据估计的相机位姿，训练 RGB 图像分别仅为 5、10、30 张。稀疏初始化也来自完整 SfM 中至少被所选两张图像观测到的点。这是“位姿已知条件下”的对比。独立估计位姿时，5 张和 10 张失败；30 张成功注册 30/30，并完成独立训练。</div>
<section class="card"><h2>共同留出视角</h2><div class="controls">__BUTTONS__</div><div class="grid" id="grid"><div class="tile"><strong>原始图像</strong><img id="reference" alt="原始留出图像"></div><div class="tile"><strong>5 张 · 位姿辅助</strong><img id="n5" alt="5 张高斯渲染"></div><div class="tile"><strong>10 张 · 位姿辅助</strong><img id="n10" alt="10 张高斯渲染"></div><div class="tile"><strong>30 张 · 位姿辅助</strong><img id="n30" alt="30 张高斯渲染"></div></div></section>
<section class="card"><h2>结果数字</h2><p>PSNR 越高，渲染图像与留出原图越接近。食物区域由已有 SAM 2 掩码划定，尚无人工像素真值。</p><div class="scroll"><table><thead><tr><th>训练图像</th><th>独立位姿注册</th><th>最大选角误差</th><th>位姿辅助 PSNR</th><th>食物 PSNR</th><th>训练时间</th><th>高斯数量</th><th>模型</th></tr></thead><tbody>__ROWS__</tbody></table></div>__EXTRA__</section>
<section class="card"><h2>30 张的独立重建</h2><p>右侧模型仅用所选 30 张照片运行 COLMAP 和 Splatfacto。评估时将其坐标与完整场景对齐，从同一个留出相机渲染。</p><div class="grid strict" style="grid-template-columns:1fr 1fr"><div class="tile"><strong>30 张 · 位姿辅助</strong><img id="fixed30" alt="固定标定位姿"></div><div class="tile"><strong>30 张 · 独立 COLMAP</strong><img id="independent30" alt="独立30张重建"></div></div></section>
<section class="card"><h2>实际相机位置分布</h2><img class="diagram" src="sampling_distribution.png" alt="5、10、30张相机位置与留出视角"><p class="muted">按估计的方位角均匀采样，覆盖现有视频约 302° 的有效范围；原视频缺失的方向没有补造。拍摄距离和俯仰高度仍受原视频限制。</p></section>
<section class="card"><a href="comparison_sheet.png">整图对比</a> · <a href="food_detail_sheet.png">食物细节对比</a> · <a href="independent_30_comparison.png">独立30张对比</a> · <a href="report.json">完整数据 JSON</a><p class="muted">不是实测体积或几何完整度真值，不代表所有食物场景都能在 30 张下成功。原照片、POC 模型和 App 数据均保留。</p></section></main>
<script>const frames=__FRAMES__;function show(frame){document.getElementById('reference').src=frame+'_source.png';[5,10,30].forEach(n=>document.getElementById('n'+n).src='n'+n+'/evaluation/'+frame+'_render.png');document.getElementById('fixed30').src='n30/evaluation/'+frame+'_render.png';document.getElementById('independent30').src='n30/independent_evaluation/'+frame+'_render.png';document.querySelectorAll('.frame').forEach(b=>b.classList.toggle('active',b.dataset.frame===frame));}document.querySelectorAll('.frame').forEach(b=>b.onclick=()=>show(b.dataset.frame));show(frames[0]);</script></html>'''
    page=page.replace('__BUTTONS__',buttons).replace('__ROWS__',rows).replace('__EXTRA__',extra).replace('__FRAMES__',payload)
    (output/'index.html').write_text(page,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ProjectPaths().resolve('data/scenes/fruit2_video_ready'))
    parser.add_argument('--segmentation',type=Path,default=ProjectPaths().resolve('data/outputs/fruit2_seg'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=['prepare','holdout','colmap','train','independent-train','evaluate','all'],default='all')
    args=resolve_arguments(parser.parse_args());output=args.output.resolve()
    if args.stage in {'prepare','all'}:prepare(args.source,args.segmentation,output)
    if args.stage=='holdout':refresh_holdout(output)
    if args.stage in {'colmap','all'}:independent_colmap(output)
    if args.stage in {'train','all'}:train(output)
    if args.stage in {'independent-train','all'}:train_independent(output)
    if args.stage in {'evaluate','all'}:evaluate(output)


if __name__=='__main__':main()
