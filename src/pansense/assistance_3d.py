"""Verified photo-to-reconstruction binding and isolated semantic rebuilds."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import shutil
from uuid import uuid4

from PIL import Image, ImageOps

from food3d.io import read_json, write_json
from food3d.paths import ProjectPaths


def normalized_photo(path: Path):
    with Image.open(path) as raw:
        picture = ImageOps.exif_transpose(raw).convert('RGB')
        picture.thumbnail((960,960))
        stream = BytesIO(); picture.save(stream, format='JPEG', quality=95)
    return stream.getvalue()


def model_binding(parent, project: Path):
    """Keep the camera/image provenance of this meal's actual reconstruction."""
    data_root = ProjectPaths(project).data_root
    saved = parent.get('reconstruction_binding') or {}
    bundle = Path(parent.get('bundle_ref') or saved.get('bundle') or '').resolve()
    allowed = data_root
    if not bundle.is_relative_to(allowed) or not (bundle/'model.json').is_file() or not (bundle/'scene.json').is_file():
        return None
    model = read_json(bundle/'model.json')
    scene = Path(model.get('source_scene') or saved.get('scene') or '').resolve()
    if not scene.is_relative_to(allowed) or not (scene/'scene.json').is_file(): return None
    frames = read_json(bundle/'scene.json')['frames']
    source = Path((parent.get('input_file') if parent.get('source_kind') in {'photo','assist-session'} else None) or parent.get('source_photo') or parent.get('input_file') or '').resolve()
    if not source.is_file() or not source.is_relative_to(allowed): return None
    reference = next((f for f in frames if (bundle/f['image_path']).resolve()==source),None)
    if reference is None and saved.get('reference_frame'):
        reference = next((f for f in frames if f['frame_id']==saved['reference_frame']),None)
    # Older photo runs already have a semantic bundle but predate explicit
    # reconstruction_binding. Its own reference is safe only after the same
    # normalized source hash check below, never by filename alone.
    if reference is None and parent.get('source_sha256') and (bundle/'semantics/instances.json').is_file():
        reference_id=read_json(bundle/'semantics/instances.json').get('reference_frame')
        reference=next((f for f in frames if f['frame_id']==reference_id),None)
    if reference is None: return None
    frame_photo = (bundle/reference['image_path']).resolve()
    original = next((f for f in read_json(scene/'scene.json')['frames'] if f['frame_id']==reference['frame_id']),None)
    if original is None or (scene/original['image_path']).resolve()!=frame_photo or not frame_photo.is_file(): return None
    expected = sha256(normalized_photo(frame_photo)).hexdigest()
    if source != frame_photo and (parent.get('source_sha256')!=expected or sha256(source.read_bytes()).hexdigest()!=expected):
        return None
    if saved and (saved.get('model_sha256')!=model['model_sha256'] or saved.get('source_sha256')!=expected): return None
    return {'scene':str(scene),'bundle':str(bundle),'reference_frame':reference['frame_id'],
            'source_sha256':expected,'model_sha256':model['model_sha256']}


def source_binding(parent, project: Path):
    data_root = ProjectPaths(project).data_root
    binding = model_binding(parent, project)
    if binding: return binding
    # Compatibility for photos uploaded through the old comparison reference.
    # An explicit model binding must never silently switch to that other model.
    if parent.get('reconstruction_binding') or parent.get('bundle_ref'): return None
    scene = data_root/'outputs/fruit2_view_counts_run01/n30/independent_colmap_scene'
    bundle = data_root/'outputs/fruit2_view_counts_run01/n30/independent_bundle'
    reference = scene/'images/000018.jpg'
    if not reference.is_file() or not (bundle/'model.json').is_file():
        return None
    expected = sha256(normalized_photo(reference)).hexdigest()
    source = Path(parent.get('input_file') or '')
    if (parent.get('source_sha256') != expected or not source.is_file()
            or sha256(source.read_bytes()).hexdigest() != expected):
        return None
    return {'scene':str(scene.resolve()), 'bundle':str(bundle.resolve()), 'reference_frame':'000018',
            'source_sha256':expected, 'model_sha256':read_json(bundle/'model.json')['model_sha256']}


def clone_geometry(source: Path, destination: Path):
    destination.mkdir(parents=True)
    for name in ('gaussians.npz','model.json'):
        shutil.copy2(source/name, destination/name)
    cameras = read_json(source/'scene.json')
    for frame in cameras['frames']:
        frame['image_path'] = str((source/frame['image_path']).resolve())
    write_json(destination/'scene.json', cameras)
    calibration=source/'measurements/calibration.json'
    if calibration.is_file() and read_json(calibration).get('source')=='white_paper_rectangle_multiview':
        from food3d.paper_scale import camera_hash
        value=read_json(calibration)
        if value.get('model_sha256')==read_json(destination/'model.json')['model_sha256'] and value.get('camera_sha256')==camera_hash(destination):
            (destination/'measurements').mkdir(exist_ok=True)
            for name in ['calibration.json','paper_reference.json','paper_detections.jpg']:
                path=source/'measurements'/name
                if path.is_file(): shutil.copy2(path,destination/'measurements'/name)


def build_selected_3d(jobs, meal_id, binding, snapshot):
    work = jobs.store.root/'uploads'/meal_id/f'three_d_{uuid4().hex}'
    work.mkdir(parents=True)
    log = work/'analysis.log'
    def stage(title, progress):
        jobs.store.fields(meal_id, three_d_analysis={'status':'running','stage':title,'progress':progress,'error':None})
    try:
        with jobs.gpu_lock:
            if meal_id in jobs.cancelled or not jobs.store.exists(meal_id): return
            jobs.release_gpu()
            stage('传播选定食物区域到其他视角', 10)
            selected = []
            for item in snapshot['items']:
                region = item.get('region') or {}
                path = Path(region.get('path') or '').resolve()
                if region.get('kind') != 'mask' or not path.is_relative_to((jobs.store.root/'uploads').resolve()) or not path.is_file():
                    raise ValueError('选定食物缺少有效分割掩码，请先重新识别')
                selected.append({'instance_id':item['instance_id'],'name':item['name'],'mask_path':str(path)})
            write_json(work/'seeds.json', {'instances':selected})
            seg, bundle = work/'segmentation', work/'bundle'
            jobs.command(meal_id,[str(jobs.recognition_python),'-m','food3d','segment',
                '--scene',binding['scene'],'--output',str(seg),'--reference-frame',binding['reference_frame'],
                '--seed-masks',str(work/'seeds.json')], log)
            stage('把选定区域关联到三维高斯', 55)
            clone_geometry(Path(binding['bundle']), bundle)
            if read_json(bundle/'model.json')['model_sha256'] != binding['model_sha256']:
                raise ValueError('源模型已发生变化，请刷新对比照片后重试')
            jobs.command(meal_id,[str(jobs.gs_python),'-m','food3d','fuse','--bundle',str(bundle),
                '--segmentation',str(seg),'--max-views','27','--width','384'], log)
            stage('检查三维语义与实例点击', 90)
            jobs.command(meal_id,[str(jobs.gs_python),'-m','food3d','evaluate','--bundle',str(bundle),
                                 '--output',str(work/'evaluation'),'--width','480'], log)
            report = read_json(work/'evaluation/report.json')
            fusion = read_json(bundle/'semantics/fusion.json')
            if not any(fusion['gaussians_per_instance'].get(str(i['instance_id']),0)>0 for i in selected):
                raise ValueError('未获得可靠的三维食物关联，请调整标记区域后重试')
            if report['food_click_accuracy'] < jobs.settings['minimum_click_accuracy']:
                raise ValueError('三维点击一致性不足，已保留二维结果，请调整区域后重试')
            jobs.command(meal_id,[str(jobs.gs_python),'-m','food3d','volume','--bundle',str(bundle)],log)
            write_json(work/'provenance.json', {'source_run':meal_id,'source_binding':binding,
                'selected_items':selected,'method':'selected_masks_sam2_propagation_visibility_fusion',
                'model_geometry_retrained':False,'evaluation':report})
            if meal_id in jobs.cancelled or not jobs.store.exists(meal_id): return
            jobs.store.fields(meal_id, bundle_ref=str(bundle), three_d_reason=None,
                three_d_analysis={'status':'ready','stage':'三维模型已就绪，可拖拽查看','progress':100,'error':None},
                quality={'instance_iou':report['mean_instance_iou'],'click_accuracy':report['food_click_accuracy']})
    except Exception as error:
        if meal_id not in jobs.cancelled and jobs.store.exists(meal_id):
            import traceback
            with log.open('a',encoding='utf-8') as stream: traceback.print_exc(file=stream)
            detail = str(error)
            if not detail or not ('\u4e00' <= detail[0] <= '\u9fff') or len(detail)>160:
                detail = '三维关联未完成，请重试；二维结果和原三维模型仍保留'
            jobs.store.fields(meal_id, three_d_reason=detail,
                three_d_analysis={'status':'failed','stage':'三维关联未完成','progress':0,'error':detail})
