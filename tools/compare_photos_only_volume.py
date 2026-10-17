"""Freeze the independent RGB experiment before reading comparison baselines."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime,timezone
import hashlib
from html import escape
import json
from pathlib import Path

import numpy as np

from food3d.io import read_json,write_json


def run(root,baseline_bundle,old_silhouette):
    root=Path(root).resolve()
    fresh=read_json(root/'fresh_pose_verified.json')
    no_pose=read_json(root/'no_pose.json')
    manifest=read_json(root/'input_manifest.json')
    segmentation=read_json(root/'segmentation_summary.json')
    evidence=read_json(root/'recognition.json')
    audit=read_json(root/'verified_read_audit.json')
    sensitivity=read_json(root/'sensitivity.json')
    assert len(manifest['photos'])==segmentation['photos']
    for photo in manifest['photos']:
        assert hashlib.sha256((root/photo['input_file']).read_bytes()).hexdigest()==photo['sha256']
    report={'schema_version':1,'input':'RGB photos only + known physical board layout',
        'input_photos':len(manifest['photos']),'fresh_segmentation':True,
        'old_poses_intrinsics_masks_geometry_read':False,'baseline_used_for_estimation':False,
        'recovery_note':'Fresh segmentation and image-model answers from this experiment were retained after fixing handling of an unobservable, occluded package. Camera fitting was refined using only the new board observations. No previous dataset was reused.',
        'no_pose':no_pose,'fresh_pose':fresh,'sensitivity':sensitivity,
        'audit':audit,'recognition':evidence,
        'accuracy_status':'No measured physical volume ground truth. Agreement with a previous estimate is not accuracy.'}
    # Baselines are deliberately unavailable until this independent result is saved.
    write_json(root/'report.json',report)
    locked_hash=hashlib.sha256((root/'report.json').read_bytes()).hexdigest()
    frozen_at=datetime.now(timezone.utc).isoformat()

    from food3d.volume import published_measurements
    baseline=published_measurements(Path(baseline_bundle))
    old=read_json(Path(old_silhouette))
    chosen=next(r for r in fresh['runs'] if r['views']==24 and r['resolution']==224)
    chosen_group=next(r for r in fresh['grouped_packaging_runs'] if r['resolution']==224)
    # Match semantic kinds, never recycled numeric instance IDs.
    mouse_ids={o['instance_id'] for o in evidence['objects'] if o['kind']=='non_food' and 'mouse' in o['label'].lower()}
    mouse=next(o for o in chosen['instances'] if o['instance_id'] in mouse_ids)
    group=chosen_group['instances'][0]
    old_volumes=old['runs'][0]['instances']
    comparisons=[]
    for name,new,old_id in [('鼠标',mouse,2),('两包零食合并外轮廓',group,1)]:
        gaussian=next(o for o in baseline['instances'] if o['instance_id']==old_id)
        silhouette=next(o for o in old_volumes if o['instance_id']==old_id)
        comparisons.append({'name':name,'fresh_photos_ml':new['volume_ml'],
            'previous_silhouette_ml':silhouette['volume_ml'],'app_gaussian_ml':gaussian['volume_ml'],
            'relative_to_previous_silhouette_pct':100*(new['volume_ml']/silhouette['volume_ml']-1),
            'relative_to_app_gaussian_pct':100*(new['volume_ml']/gaussian['volume_ml']-1)})
    diagnostics=fresh['pose_diagnostics'];held=set(fresh['calibration']['calibration_heldout_frames'])
    quality={'usable_cameras':len(diagnostics),'intrinsics_training_views':fresh['calibration']['views'],
        'intrinsics_reprojection_rms_px':fresh['calibration']['reprojection_error'],
        'heldout_marker_median_px':float(np.median([d['heldout_marker_rms_px'] for d in diagnostics])),
        'heldout_marker_p90_px':float(np.percentile([d['heldout_marker_rms_px'] for d in diagnostics],90)),
        'usable_intrinsics_heldout_photos':sum(d['frame_id'] in held for d in diagnostics),
        'intrinsics_heldout_photo_count':len(held)}
    comparison={'independent_result_sha256':locked_hash,'independent_result_saved_at_utc':frozen_at,
        'baselines_read_after_independent_result_saved':True,'comparisons':comparisons,'quality':quality,
        'warning':'The two packages are heavily occluded. Per-object relaxed visual hulls can overlap; their volumes must not be added. Combined outer silhouette is the comparison quantity.'}
    write_json(root/'comparison.json',comparison)

    rows=[]
    for instance in chosen['instances']:
        prior=next(o for o in no_pose['instances'] if o['instance_id']==instance['instance_id'])
        value=prior['volume_ml']
        rough='无法可靠估算' if value is None else f'{value:.1f} mL（先验范围 {prior["prior_sensitivity_ml"][0]:.1f}–{prior["prior_sensitivity_ml"][1]:.1f}）'
        rows.append(f'<tr><td>{escape(instance["name"])}</td><td>{rough}</td><td>{instance["volume_ml"]:.1f} mL</td></tr>')
    compare_rows=''.join(f'<tr><td>{o["name"]}</td><td>{o["fresh_photos_ml"]:.1f}</td><td>{o["previous_silhouette_ml"]:.1f}</td><td>{o["app_gaussian_ml"]:.1f}</td><td>{o["relative_to_previous_silhouette_pct"]:+.2f}%</td></tr>' for o in comparisons)
    stability=''.join('<tr><td>'+str(r['views'])+'</td><td>'+str(r['resolution'])+'</td>'+''.join(f'<td>{o["volume_ml"]:.1f}</td>' for o in r['instances'])+'</tr>' for r in fresh['runs'])
    sensitivity_rows=''.join(f'<tr><td>{r["agreement"]:.0%}</td><td>{r["dilation_pixels"]}</td>'+''.join(f'<td>{o["volume_ml"]:.1f}</td>' for o in r['instances'])+'</tr>' for r in [sensitivity['reference'],*sensitivity['tests']])
    picture=base64.b64encode((root/'fresh_mask_sheet.png').read_bytes()).decode('ascii')
    assumptions=''.join(f'<li><b>{escape(o["name"])}</b>：{escape(o["shape_prior"]["assumptions"])}</li>' for o in no_pose['instances'])
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>照片独立体积实验</title><style>body{{font:16px/1.7 "Microsoft YaHei",sans-serif;color:#183d3b;background:#edf2ed;margin:0}}main{{max-width:1040px;margin:40px auto;padding:28px;background:#fff;border-radius:20px}}h1,h2{{line-height:1.3}}table{{border-collapse:collapse;width:100%;margin:20px 0}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #d7e2dc}}th{{background:#edf4ef}}.note{{padding:18px;background:#fff4d8;border-radius:12px}}img{{max-width:100%}}code{{overflow-wrap:anywhere}}details{{margin:24px 0}}small{{color:#61716b}}a{{color:#17615a}}</style>
<main><h1>从照片开始的两条独立体积流程</h1>
<p>输入 {len(manifest['photos'])} 张原始照片及定位板已知尺寸。重新完成识别、分割及相机估计。没有读取旧位姿、旧内参、旧掩码、高斯点云或网格。实验没有接入 App。</p>
<p class="note"><b>结论：</b>完全不使用位姿时，厚度只能借助模型先验，当前照片不足以可靠估计全部对象。允许从照片重新估计位姿时，流程可以独立跑通；结果仍是有遮挡和分割误差的外形粗估，并未通过实测体积验证。</p>
<h2>两种流程的结果</h2><p>单位 mL = cm³。重新估计位姿结果采用预先设定的 24 视角、224 网格；无位姿的范围仅表示形状假设变化，不能当作误差条或置信区间。</p>
<table><tr><th>对象</th><th>完全无位姿：形状先验粗估</th><th>照片重新估计位姿：轮廓体积</th></tr>{''.join(rows)}
<tr><td>两包零食合并外轮廓</td><td>下层被遮挡，不能给出可信合计</td><td>{group['volume_ml']:.1f} mL</td></tr></table>
<p class="note">重新识别出了鼠标与上下两包包装，共 3 个对象；旧大模型结果把两包合并为 1 项。逐包的遮挡容忍轮廓体积可能在空间中重叠，不能相加，尤其不能把下层包装的数值当作准确体积。合并外轮廓作为单独的比较量计算。</p>
<h2>与以前结果比较</h2><table><tr><th>相同对象范围</th><th>本次完整独立流程 mL</th><th>旧轮廓流程 mL</th><th>App 高斯凸包 mL</th><th>相对旧轮廓</th></tr>{compare_rows}</table>
<p>旧结果只在独立估计保存并计算 SHA-256 后读取。两种估计接近不能证明体积准确。</p>
<h2>重新标定与独立校验</h2><p>仅用照片中的 3 cm 标记与 20 × 20 cm 定位板重新标定焦距和相机位置。内参标定采用 {quality['intrinsics_training_views']} 个视角，重投影误差 {quality['intrinsics_reprojection_rms_px']:.2f} px；有 {quality['usable_cameras']} 张照片通过完整质量筛选。预留的 {quality['intrinsics_heldout_photo_count']} 张内参验证照片中，{quality['usable_intrinsics_heldout_photos']} 张通过筛选。</p>
<p>每张照片再留出一个标记，不参与该张照片的位姿拟合；通过筛选照片的留出标记误差中位数 {quality['heldout_marker_median_px']:.2f} px，90 分位 {quality['heldout_marker_p90_px']:.2f} px。它验证投影几何一致性，不验证隐藏表面或真实体积。</p>
<details open><summary>视角数和网格精度的稳定性（mL）</summary><table><tr><th>视角数</th><th>网格</th><th>鼠标</th><th>上层包装</th><th>下层包装</th></tr>{stability}</table></details>
<details open><summary>轮廓容差的敏感性（24 视角、160 网格，mL）</summary><table><tr><th>轮廓一致性要求</th><th>掩码扩张 px</th><th>鼠标</th><th>上层包装</th><th>下层包装</th></tr>{sensitivity_rows}</table><p>容差改变会改变体积，说明精细网格收敛不等于真实精度已被验证。没有通过调整参数去匹配旧体积。</p></details>
<h2>全新识别与分割</h2><p>紫色为鼠标，蓝色为上层包装，橙色为下层包装；白色表示两个对象间的归属不确定区域。所有掩码均由本次照片重新生成。</p><img src="data:image/png;base64,{picture}" alt="六张新关键帧的分割结果">
<details><summary>完全无位姿流程的形状假设</summary><p>2D 定位板单应变换只给出投影到板面上的可见面积及短边；厚度比例和体积填充系数来自仅看照片的模型推断。采用 面积 × 假设高度 × 假设填充系数，未恢复隐藏轮廓。</p><ul>{assumptions}</ul><p>平面单应变换不能把高出板面的点当作板面点来准确测量，物体高度、遮挡和模型先验都可能造成系统偏差。</p></details>
<h2>数据与复现</h2><p><a href="report.json">冻结的独立实验结果</a> · <a href="comparison.json">后读取的基线比较</a> · <a href="input_manifest.json">原始照片校验和</a> · <a href="verified_read_audit.json">独立阶段读文件审计</a></p>
<p><code>python -m food3d.photos_only_volume --input 原始照片目录 --output 新实验目录</code></p>
<small>独立估计 SHA-256：{locked_hash}<br>假设定位板平整且按原尺寸打印；没有物理体积真值，包装测量包含空气，无法直接推算食物净体积或重量。</small></main></html>'''
    (root/'report.html').write_text(html,encoding='utf-8')
    print(json.dumps({'comparison':comparisons,'quality':quality,'report':str(root/'report.html')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--baseline-bundle',type=Path,required=True)
    parser.add_argument('--old-silhouette',type=Path,required=True)
    args=parser.parse_args();run(args.experiment,args.baseline_bundle,args.old_silhouette)
