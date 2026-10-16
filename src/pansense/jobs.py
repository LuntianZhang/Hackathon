"""Serialized GPU jobs, with local artifacts and safe 2D failure fallback."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, CancelledError
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time
from uuid import uuid4

from PIL import Image
import yaml

from food3d.io import read_json, write_json, load_config
from food3d.paths import ProjectPaths
from food3d.schemas import load_scene

class Jobs:
    def __init__(self, store, project: Path, gpu_lock: threading.Lock, enabled=True, release_gpu=None):
        self.store, self.project, self.gpu_lock = store, project.resolve(), gpu_lock
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pansense-analysis')
        self.enabled, self.futures, self.processes = enabled, {}, {}
        self.cancelled = set()
        self.release_gpu = release_gpu or (lambda: None)
        self.paths = ProjectPaths(self.project)
        self.recognition_python = self.paths.recognition_python
        self.gs_python = self.paths.reconstruction_python
        self.settings = read_yaml(self.project/'configs/pansense.yaml')
        if not 100 <= int(self.settings['iterations']) <= 30000:
            raise ValueError('训练步数必须在 100 至 30000 之间')
        if not 128 <= int(self.settings['fusion_width']) <= 960:
            raise ValueError('融合分辨率必须在 128 至 960 之间')

    def recover(self):
        for meal in self.store.list(include_drafts=True):
            if meal.get('three_d_analysis',{}).get('status') in {'queued','running'}:
                self.store.fields(meal['id'], three_d_analysis={'status':'failed','stage':'三维关联已中断',
                    'progress':0,'error':'服务已重启，可重新生成三维关联'})
            if meal['analysis']['status'] in {'queued', 'running'}:
                self.store.fields(meal['id'], analysis={'status':'failed', 'stage':'上次分析已中断',
                  'progress':0, 'error':'应用服务重启，照片和已有清单仍保留，可重新分析'})

    def submit(self, meal_id, kind, input_path):
        self.cancelled.discard(meal_id)
        if kind in {'video-prepare','scan-prepare'}: self.store.fields(meal_id, video_preparing=True)
        self.store.fields(meal_id, analysis={'status':'queued','stage':'等待分析','progress':0,'error':None})
        if self.enabled:
            self.futures[meal_id] = self.pool.submit(self.run, meal_id, kind, Path(input_path))

    def submit_3d(self, meal_id, binding):
        from .assistance_3d import build_selected_3d
        value = self.store.get(meal_id)
        if value.get('three_d_analysis',{}).get('status') in {'queued','running'} or value.get('bundle_ref'):
            return
        if not value['items']:
            raise ValueError('该版本没有可靠食物实例，请先调整识别结果')
        self.cancelled.discard(meal_id)
        self.store.fields(meal_id, three_d_analysis={'status':'queued','stage':'等待生成三维关联','progress':0,'error':None})
        if self.enabled:
            self.futures[meal_id] = self.pool.submit(build_selected_3d, self, meal_id, binding, value)

    def stage(self, meal_id, title, progress, **extra):
        self.store.fields(meal_id, analysis={'status':'running','stage':title,'progress':progress,'error':None}, **extra)

    def command(self, meal_id, argv: list[str], log: Path, timeout=1800, progress_range=None):
        if meal_id in self.cancelled:
            raise RuntimeError('分析已停止')
        environment = self.paths.environment()
        with log.open('a', encoding='utf-8') as stream:
            stream.write('\nCOMMAND '+json.dumps(argv, ensure_ascii=False)+'\n')
            stream.flush()
            process = subprocess.Popen(argv, cwd=self.project, stdout=stream, stderr=subprocess.STDOUT, env=environment)
            self.processes[meal_id] = process
            started = time.monotonic()
            last_progress_at, last_progress = started, -1
            try:
                while process.poll() is None:
                    if meal_id in self.cancelled or not self.store.exists(meal_id):
                        process.terminate()
                        raise RuntimeError('餐食已删除，停止分析')
                    if time.monotonic()-started > timeout:
                        process.terminate()
                        raise RuntimeError('分析超时，可先使用食物清单保存这餐')
                    if progress_range and time.monotonic()-last_progress_at > 3:
                        import re
                        with log.open('rb') as reader:
                            reader.seek(max(0,log.stat().st_size-8192))
                            tail=reader.read().decode('utf-8',errors='replace')
                        matches=re.findall(r'\b\d+ \(([\d.]+)%\)',tail)
                        if matches:
                            low,high=progress_range
                            progress=int(low+(high-low)*float(matches[-1])/100)
                            if progress!=last_progress:
                                self.stage(meal_id,'生成可旋转的三维餐盘',progress)
                                last_progress=progress
                        last_progress_at=time.monotonic()
                    time.sleep(.3)
                if process.returncode:
                    raise RuntimeError('分析未完成，请先使用食物清单；详细诊断保存在本机日志')
            finally:
                if process.poll() is None:
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: process.kill()
                self.processes.pop(meal_id, None)

    def publish_items(self, meal_id, entries, photo, **extra):
        def publish(value):
            # Never silently overwrite manual corrections or a record saved
            # through the 2D/manual fallback while recognition was running.
            if value['saved_at'] and not self.store.profile()['retain_media']:
                return
            if not value['items'] and not value['saved_at']:
                value['items'] = entries
            value.update(source_photo=str(photo), **extra)
        self.store.mutate(meal_id, publish)

    def run(self, meal_id, kind, input_path):
        work = self.store.root/'uploads'/meal_id/f'analysis_{uuid4().hex}'
        work.mkdir(parents=True)
        log = work/'analysis.log'
        try:
            if kind == 'photo' and self.store.get(meal_id).get('assistance',{}).get('mode') == 'none':
                if meal_id in self.cancelled or not self.store.exists(meal_id): return
                self.publish_items(meal_id, [], input_path,
                    semantic_summary={'mode':'none','instance_count':0})
                self.store.fields(meal_id, analysis={'status':'ready','stage':'照片已保留，请填写食物清单',
                    'progress':100,'error':None}, three_d_reason='未使用语义分割，可手工填写食物清单')
                return
            with self.gpu_lock:
                if meal_id in self.cancelled or not self.store.exists(meal_id): return
                self.release_gpu()
                if kind == 'photo':
                    self.stage(meal_id,'识别食物与视觉区域',25)
                    assistance = self.store.get(meal_id).get('assistance')
                    arguments = []
                    if assistance:
                        write_json(work/'assistance.json', assistance)
                        arguments = ['--assistance', str(work/'assistance.json')]
                    self.command(meal_id, [str(self.recognition_python), '-m','pansense.photo',
                        '--input',str(input_path),'--output',str(work/'photo'),
                        '--catalog',str(self.project/'app/catalog/foods.json'), *arguments], log, 600)
                    result = read_json(work/'photo/recognition.json')
                    entries = [self.store.new_item(entry['name'], entry['catalog_key'], entry['instance_id'],
                        source='manual_prompt' if entry.get('name_source') == 'manual_prompt' else 'automatic',
                        region={'kind':'mask','path':entry['mask_path']},
                        recognition_score=entry['recognition_score'],
                        recognition_margin=entry.get('recognition_margin'),
                        name_source=entry.get('name_source','automatic'),
                        needs_review=entry.get('needs_review',True)) for entry in result['items']]
                    self.publish_items(meal_id, entries, result['photo'], recognition_result=str(work/'photo/recognition.json'),
                        semantic_summary={'mode':(assistance or {}).get('mode','auto'),
                            'instance_count':len(entries),'elapsed_seconds':result.get('elapsed_seconds'),
                            'missing_hints':result.get('missing_hints',[])})
                    title = '分割完成，名称按人工提示保留' if assistance and assistance['mode']=='manual' else '识别完成，请核对食物名称'
                    self.store.fields(meal_id, analysis={'status':'ready','stage':title,
                      'progress':100,'error':None if entries else '未找到可靠候选，请手工添加食物'},
                      three_d_reason='单张照片缺少多视角信息，可继续使用二维食物清单记录')
                else:
                    self.video(meal_id, input_path, work, log, prepare_only=kind in {'video-prepare','scan-prepare'},from_photos=kind in {'scan','scan-prepare'})
            # Queue after releasing the shared GPU lock. The recognition result
            # and its masks are the input to this meal's geometry, not a second
            # independent recognition of a canned demo.
            if kind == 'photo' and meal_id not in self.cancelled and self.store.exists(meal_id):
                from .assistance_3d import source_binding
                value = self.store.get(meal_id)
                binding = source_binding(value, self.project, self.store.root/'uploads')
                if binding and value['items'] and not (value['saved_at'] and not self.store.profile()['retain_media']):
                    self.store.fields(meal_id,reconstruction_binding=binding,semantic_auto_3d=True,
                        three_d_reason='正在将本次食物分割应用到对应的三维餐盘')
                    self.submit_3d(meal_id,binding)
        except Exception as error:
            if meal_id not in self.cancelled and self.store.exists(meal_id):
                import traceback
                with log.open('a',encoding='utf-8') as stream: traceback.print_exc(file=stream)
                detail=str(error)
                if not detail or not ('\u4e00'<=detail[0]<='\u9fff') or len(detail)>160:
                    detail='分析服务遇到问题，请稍后重试，或手工添加食物继续记录'
                self.store.fields(meal_id, analysis={'status':'failed','stage':'分析未完成','progress':0,
                    'error':detail}, three_d_reason='三维暂不可用，已有食物清单仍可修改和保存')

    def video(self, meal_id, input_path, work, log, prepare_only=False,from_photos=False):
        config = load_config(self.project/self.settings['scene_config'])
        tools = config['frames']
        if not from_photos:
            probe = subprocess.run([tools['ffprobe'],'-v','error','-show_format','-of','json',str(input_path)],
                                   capture_output=True, text=True, encoding='utf-8', check=True)
            duration = float(json.loads(probe.stdout)['format']['duration'])
            if not 3 <= duration <= 180:
                raise ValueError('请使用 3 秒至 3 分钟的环绕视频')
            config['frames']['fps'] = min(3, 119/duration)
        else:
            config['photos']={'max_frames':180,'max_long_edge':1280,'min_sharpness':12,'min_difference':.5,'jpeg_quality':95}
        config['reconstruction']['stop_split_at'] = int(self.settings['stop_split_at'])
        config_path = work/'config.yaml'
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
        scene, seg, training, bundle = [work/name for name in ['scene','segmentation','training','bundle']]
        def recognition(*args): self.command(meal_id, [str(self.recognition_python),'-m','food3d',*map(str,args)],log)
        def gs(*args): self.command(meal_id, [str(self.gs_python),'-m','food3d',*map(str,args)],log)
        prepared = self.store.get(meal_id).get('prepared_video')
        if prepared:
            scene = Path(prepared['scene']).resolve()
            if not scene.is_relative_to((self.store.root/'uploads'/meal_id).resolve()):
                raise ValueError('视频参考帧不属于当前餐盘')
        else:
            self.stage(meal_id,'导入手机扫描照片' if from_photos else '筛选视频画面',5)
            if from_photos:
                recognition('photos','--input',input_path,'--scene',scene,'--config',config_path)
            else:
                recognition('extract','--video',input_path,'--scene',scene,'--config',config_path)
            self.stage(meal_id,'关联不同拍摄角度，准备标注参考帧' if prepare_only else '关联不同拍摄角度',15)
            recognition('colmap','--scene',scene,'--config',config_path)
        _, frames = load_scene(scene/'scene.json')
        if from_photos and hasattr(self,'verify_scan'):
            self.verify_scan(self.store.get(meal_id)['scan_id'],scene,input_path)
        seed = next((f for f in frames if prepared and f.frame_id==prepared['reference_frame']),None) if prepared else frames[min(15,len(frames)-1)]
        if seed is None: raise ValueError('标注参考帧已不可用，请重新导入视频')
        if prepare_only:
            self.store.fields(meal_id,source_photo=str(scene/seed.image_path),video_preparing=False,
                prepared_video={'scene':str(scene),'reference_frame':seed.frame_id},
                analysis={'status':'ready','stage':'参考帧已准备好，请标记食物','progress':100,'error':None})
            return
        assistance = self.store.get(meal_id).get('assistance')
        no_semantics = assistance and assistance['mode']=='none'
        entries = []
        if not no_semantics:
            self.stage(meal_id,'识别与跟踪食物',30)
            extra_args = []
            if assistance:
                write_json(work/'assistance.json', assistance)
                extra_args = ['--assistance', work/'assistance.json']
            recognition('segment','--scene',scene,'--output',seg,'--reference-frame',seed.frame_id,*extra_args)
            instances = read_json(seg/'instances.json')['instances']
            manual = assistance and assistance['mode']=='manual'
            entries = [self.store.new_item(entry['name'], instance_id=entry['instance_id'], source='manual_prompt' if manual else 'automatic',
                 region={'kind':'mask','path':str(seg/'masks'/f'{seed.frame_id}.png'),'instance_id':entry['instance_id']},
                 needs_review=not manual) for entry in instances if entry['instance_id'] > 0]
        self.publish_items(meal_id,entries,scene/seed.image_path,
            semantic_summary={'mode':(assistance or {}).get('mode','auto'),'instance_count':len(entries)})
        self.stage(meal_id,'生成可旋转的三维餐盘',45)
        # Run the worker directly, so cancellation owns the actual GPU process.
        self.command(meal_id,[str(self.gs_python),'-m','food3d.reconstruction.train_worker','--scene',str(scene),
            '--output',str(training),'--iterations',str(self.settings['iterations']),'--config',str(config_path)],log,progress_range=(45,78))
        checkpoint = max(training.rglob('*.ckpt'),key=lambda path:path.stat().st_mtime)
        transform = checkpoint.parent.parent/'dataparser_transforms.json'
        self.stage(meal_id,'导出三维餐盘' if no_semantics else '关联三维食物区域',80)
        gs('export','--checkpoint',checkpoint,'--scene',scene,'--output',bundle,'--transform',transform)
        if no_semantics:
            def complete_geometry(value):
                if value['saved_at'] and not self.store.profile()['retain_media']: return
                value.update(bundle_ref=str(bundle), analysis={'status':'ready','stage':'三维外观已完成，可手工补充食物',
                    'progress':100,'error':None}, three_d_reason=None, quality=None)
            self.store.mutate(meal_id,complete_geometry)
            return
        gs('fuse','--bundle',bundle,'--segmentation',seg,'--max-views',self.settings['fusion_max_views'],'--width',self.settings['fusion_width'])
        self.stage(meal_id,'检查三维关联效果',95)
        gs('evaluate','--bundle',bundle,'--output',work/'evaluation')
        report = read_json(work/'evaluation/report.json')
        if report['mean_instance_iou'] < self.settings['minimum_instance_iou'] or report['food_click_accuracy'] < self.settings['minimum_click_accuracy']:
            raise RuntimeError('三维关联效果不足，已保留二维识别清单供记录')
        self.stage(meal_id,'用白色纸片标定体积',97)
        paper=self.settings.get('paper_reference',{})
        if paper.get('enabled',True) and self.store.get(meal_id).get('paper_present',True):
            gs('paper-scale','--bundle',bundle,'--length-cm',paper.get('length_cm',10),'--width-cm',paper.get('width_cm',2))
        gs('volume','--bundle',bundle)
        def complete(value):
            if value['saved_at'] and not self.store.profile()['retain_media']: return
            value.update(bundle_ref=str(bundle),analysis={'status':'ready','stage':'三维餐盘已完成',
                 'progress':100,'error':None}, three_d_reason=None,
                 quality={'instance_iou':report['mean_instance_iou'],'click_accuracy':report['food_click_accuracy']})
        self.store.mutate(meal_id,complete)

    def cancel(self, meal_id):
        self.cancelled.add(meal_id)
        future = self.futures.get(meal_id)
        if future: future.cancel()
        process = self.processes.get(meal_id)
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill()
        # Asset deletion must wait for the worker to finish writing its log/results.
        if future and not future.cancelled():
            try: future.result(timeout=15)
            except CancelledError: pass

    def close(self):
        for meal_id in list(self.processes): self.cancel(meal_id)
        self.pool.shutdown(wait=True, cancel_futures=True)

def read_yaml(path):
    return yaml.safe_load(path.read_text(encoding='utf-8'))
