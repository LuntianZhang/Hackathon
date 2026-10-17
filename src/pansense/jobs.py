"""Serialized GPU jobs, with local artifacts and safe 2D failure fallback."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, CancelledError
from contextlib import contextmanager, ExitStack
import json
import os
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


def terminate_worker(process):
    """Stop this worker and its Agent child before deleting owned media."""
    if process.poll() is not None: return
    if os.name == 'nt':
        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False,
            creationflags=subprocess.CREATE_NO_WINDOW)
    if process.poll() is None: process.terminate()

class Jobs:
    def __init__(self, store, project: Path, gpu_lock: threading.Lock, enabled=True, release_gpu=None):
        self.store, self.project, self.gpu_lock = store, project.resolve(), gpu_lock
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='pansense-analysis')
        self.enabled, self.futures, self.processes = enabled, {}, {}
        self.cancelled = set()
        self._lock = threading.RLock()
        self._events = {}
        self._pending = {}
        self._current = threading.local()
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
        with self._lock:
            event = self._new_job(meal_id)
            if kind in {'video-prepare','scan-prepare'}: self.store.fields(meal_id, video_preparing=True)
            self.store.fields(meal_id, analysis={'status':'queued','stage':'等待分析','progress':0,'error':None})
            if self.enabled:
                self._queue(meal_id, event, self.run, meal_id, kind, Path(input_path))

    def submit_3d(self, meal_id, binding):
        from .assistance_3d import build_selected_3d
        with self._lock:
            # Auto reconstruction belongs to the same operation as recognition.
            if getattr(self._current, 'event', None) is not None: self.check_cancelled(meal_id)
            value = self.store.get(meal_id)
            if value.get('three_d_analysis',{}).get('status') in {'queued','running'} or value.get('bundle_ref'):
                return
            if not value['items']:
                raise ValueError('该版本没有可靠食物实例，请先调整识别结果')
            event = self._new_job(meal_id, continuation=True)
            self.store.fields(meal_id, three_d_analysis={'status':'queued','stage':'等待生成三维关联','progress':0,'error':None})
            if self.enabled:
                self._queue(meal_id, event, build_selected_3d, self, meal_id, binding, value)

    def _queue(self, meal_id, event, function, *args):
        future = self.pool.submit(self._execute, event, function, *args)
        self.futures[meal_id] = future
        self._pending.setdefault(meal_id, set()).add(future)
        def finished(done):
            with self._lock: self._pending[meal_id].discard(done)
        future.add_done_callback(finished)

    def _new_job(self, meal_id, continuation=False):
        current = getattr(self._current, 'event', None)
        if continuation and current is self._events.get(meal_id) and current is not None:
            self.check_cancelled(meal_id)
            return current
        previous = self.futures.get(meal_id)
        if previous and not previous.done() and not self._events[meal_id].is_set():
            raise ValueError('这餐正在分析，请先中断再重新分析')
        event = self._events[meal_id] = threading.Event()
        self.cancelled.discard(meal_id)
        return event

    def _execute(self, event, function, *args):
        self._current.event = event
        try:
            if not event.is_set(): function(*args)
        except CancelledError:
            pass
        finally:
            del self._current.event

    def is_cancelled(self, meal_id):
        event = getattr(self._current, 'event', None)
        return event.is_set() if event is not None else meal_id in self.cancelled

    def check_cancelled(self, meal_id):
        if self.is_cancelled(meal_id) or not self.store.exists(meal_id):
            raise CancelledError('分析已中断')

    def mutate(self, meal_id, function):
        # Stop and publication share a lock so a late result cannot undo a stop.
        with self._lock:
            self.check_cancelled(meal_id)
            return self.store.mutate(meal_id, function)

    def fields(self, meal_id, **fields):
        return self.mutate(meal_id, lambda value: value.update(fields))

    @contextmanager
    def use_gpu(self, meal_id):
        while not self.gpu_lock.acquire(timeout=.2): self.check_cancelled(meal_id)
        try:
            self.check_cancelled(meal_id)
            yield
        finally:
            self.gpu_lock.release()

    def stage(self, meal_id, title, progress, **extra):
        self.fields(meal_id, analysis={'status':'running','stage':title,'progress':progress,'error':None}, **extra)

    def command(self, meal_id, argv: list[str], log: Path, timeout=1800, progress_range=None, output=None):
        self.check_cancelled(meal_id)
        environment = self.paths.environment()
        with ExitStack() as stack:
            stream = stack.enter_context(log.open('a', encoding='utf-8'))
            stdout = stack.enter_context(output.open('w', encoding='utf-8')) if output else stream
            stream.write('\nCOMMAND '+json.dumps(argv, ensure_ascii=False)+'\n')
            stream.flush()
            with self._lock:
                self.check_cancelled(meal_id)
                process = subprocess.Popen(argv, cwd=self.project, stdout=stdout, stderr=stream, env=environment)
                self.processes[meal_id] = process
            started = time.monotonic()
            last_progress_at, last_progress = started, -1
            try:
                while process.poll() is None:
                    if self.is_cancelled(meal_id) or not self.store.exists(meal_id):
                        terminate_worker(process)
                        raise CancelledError('分析已中断')
                    if time.monotonic()-started > timeout:
                        terminate_worker(process)
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
                self.check_cancelled(meal_id)
                if process.returncode:
                    raise RuntimeError('分析未完成，请先使用食物清单；详细诊断保存在本机日志')
            finally:
                if process.poll() is None:
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: process.kill()
                with self._lock:
                    if self.processes.get(meal_id) is process: self.processes.pop(meal_id, None)

    def publish_items(self, meal_id, entries, photo, **extra):
        def publish(value):
            # Never silently overwrite manual corrections or a record saved
            # through the 2D/manual fallback while recognition was running.
            if value['saved_at'] and not self.store.profile()['retain_media']:
                return
            if not value['items'] and not value['saved_at']:
                value['items'] = entries
            value.update(source_photo=str(photo), **extra)
        self.mutate(meal_id, publish)

    def run(self, meal_id, kind, input_path):
        work = self.store.root/'uploads'/meal_id/f'analysis_{uuid4().hex}'
        work.mkdir(parents=True)
        log = work/'analysis.log'
        try:
            if kind == 'photo' and self.store.get(meal_id).get('assistance',{}).get('mode') == 'none':
                self.check_cancelled(meal_id)
                self.publish_items(meal_id, [], input_path,
                    semantic_summary={'mode':'none','instance_count':0})
                self.fields(meal_id, analysis={'status':'ready','stage':'照片已保留，请填写食物清单',
                    'progress':100,'error':None}, three_d_reason='未使用语义分割，可手工填写食物清单')
                return
            with self.use_gpu(meal_id):
                self.release_gpu()
                if kind == 'photo':
                    self.stage(meal_id,'识别食物与视觉区域',25)
                    assistance = self.store.get(meal_id).get('assistance')
                    arguments = []
                    if assistance:
                        write_json(work/'assistance.json', assistance)
                        arguments = ['--assistance', str(work/'assistance.json')]
                    if assistance and assistance.get('mode') == 'llm':
                        from .assistance_3d import source_binding
                        binding = source_binding(self.store.get(meal_id), self.project, self.store.root/'uploads')
                        if binding:
                            self.stage(meal_id,'大模型联合识别多个视角并跟踪食物',25)
                            arguments += ['--scene',binding['scene'],'--reference-frame',binding['reference_frame']]
                    self.command(meal_id, [str(self.recognition_python), '-m','pansense.photo',
                        '--input',str(input_path),'--output',str(work/'photo'),
                        '--catalog',str(self.project/'app/catalog/foods.json'), *arguments], log,
                        1800 if assistance and assistance.get('mode')=='llm' else 600)
                    result = read_json(work/'photo/recognition.json')
                    entries = [self.store.new_item(entry['name'], entry['catalog_key'], entry['instance_id'],
                        source='manual_prompt' if entry.get('name_source') == 'manual_prompt' else 'automatic',
                        region={'kind':'mask','path':entry['mask_path']},
                        recognition_score=entry['recognition_score'],
                        recognition_margin=entry.get('recognition_margin'),
                        name_source=entry.get('name_source','automatic'),
                        object_kind=entry.get('kind'),
                        needs_review=entry.get('needs_review',True)) for entry in result['items']]
                    self.publish_items(meal_id, entries, result['photo'], recognition_result=str(work/'photo/recognition.json'),
                        semantic_summary={'mode':(assistance or {}).get('mode','auto'),
                            'instance_count':len(entries),'elapsed_seconds':result.get('elapsed_seconds'),
                            'recognition_views':len(result.get('keyframes',[])), 'model':result.get('model'),
                            'missing_hints':result.get('missing_hints',[])})
                    title = '分割完成，名称按人工提示保留' if assistance and assistance['mode']=='manual' else '识别完成，请核对食物名称'
                    self.fields(meal_id, analysis={'status':'ready','stage':title,
                      'progress':100,'error':None if entries else '未找到可靠候选，请手工添加食物'},
                      three_d_reason='单张照片缺少多视角信息，可继续使用二维食物清单记录')
                else:
                    self.video(meal_id, input_path, work, log, prepare_only=kind in {'video-prepare','scan-prepare'},from_photos=kind in {'scan','scan-prepare'})
            # Queue after releasing the shared GPU lock. The recognition result
            # and its masks are the input to this meal's geometry, not a second
            # independent recognition of a canned demo.
            if kind == 'photo' and not self.is_cancelled(meal_id) and self.store.exists(meal_id):
                from .assistance_3d import source_binding
                value = self.store.get(meal_id)
                binding = source_binding(value, self.project, self.store.root/'uploads')
                if binding and value['items'] and not (value['saved_at'] and not self.store.profile()['retain_media']):
                    self.fields(meal_id,reconstruction_binding=binding,semantic_auto_3d=True,
                        three_d_reason='正在将本次食物分割应用到对应的三维餐盘')
                    self.submit_3d(meal_id,binding)
        except CancelledError:
            pass
        except Exception as error:
            if not self.is_cancelled(meal_id) and self.store.exists(meal_id):
                import traceback
                with log.open('a',encoding='utf-8') as stream: traceback.print_exc(file=stream)
                detail=str(error)
                if not detail or not ('\u4e00'<=detail[0]<='\u9fff') or len(detail)>160:
                    detail='分析服务遇到问题，请稍后重试，或手工添加食物继续记录'
                self.fields(meal_id, analysis={'status':'failed','stage':'分析未完成','progress':0,
                    'error':detail}, three_d_reason='三维暂不可用，已有食物清单仍可修改和保存')

    def video(self, meal_id, input_path, work, log, prepare_only=False,from_photos=False):
        config = load_config(self.project/self.settings['scene_config'])
        tools = config['frames']
        prepared = self.store.get(meal_id).get('prepared_video')
        if not from_photos and not prepared:
            probe = work/'probe.json'
            self.command(meal_id,[tools['ffprobe'],'-v','error','-show_format','-of','json',str(input_path)],
                         log, timeout=30, output=probe)
            duration = float(read_json(probe)['format']['duration'])
            if not 3 <= duration <= 180:
                raise ValueError('请使用 3 秒至 3 分钟的环绕视频')
            config['frames']['fps'] = min(3, 119/duration)
        elif from_photos:
            config['photos']={'max_frames':180,'max_long_edge':1280,'min_sharpness':12,'min_difference':.5,'jpeg_quality':95}
        config['reconstruction']['stop_split_at'] = int(self.settings['stop_split_at'])
        config_path = work/'config.yaml'
        config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding='utf-8')
        scene, seg, training, bundle = [work/name for name in ['scene','segmentation','training','bundle']]
        def recognition(*args): self.command(meal_id, [str(self.recognition_python),'-m','food3d',*map(str,args)],log)
        def gs(*args): self.command(meal_id, [str(self.gs_python),'-m','food3d',*map(str,args)],log)
        def publish_background():
            background=scene/'board_background/report.json'
            if background.is_file():
                report=read_json(background)
                self.fields(meal_id,background_summary={key:report[key] for key in
                    ('enabled','status','processed_frames','excluded_frames')},
                    background_preview=str(scene/'board_background/debug/preview.jpg'))
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
            if self.store.get(meal_id).get('board_background'):
                self.stage(meal_id,'跟随定位板跟踪完整物体，排除背景',10)
                recognition('board-background','--scene',scene,'--minimum-frames',config['colmap'].get('minimum_registered_frames',8))
                publish_background()
            self.stage(meal_id,'关联不同拍摄角度，准备标注参考帧' if prepare_only else '关联不同拍摄角度',15)
            recognition('colmap','--scene',scene,'--config',config_path)
        publish_background()
        _, frames = load_scene(scene/'scene.json')
        if from_photos and hasattr(self,'verify_scan'):
            self.verify_scan(self.store.get(meal_id)['scan_id'],scene,input_path)
        seed = next((f for f in frames if prepared and f.frame_id==prepared['reference_frame']),None) if prepared else frames[min(15,len(frames)-1)]
        if seed is None: raise ValueError('标注参考帧已不可用，请重新导入视频')
        if prepare_only:
            self.fields(meal_id,source_photo=str(scene/seed.image_path),video_preparing=False,
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
                 name_source=entry.get('name_source','automatic'),object_kind=entry.get('kind'),
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
            self.mutate(meal_id,complete_geometry)
            return
        if self.settings.get('board_reference',{}).get('enabled',True):
            self.stage(meal_id,'用定位板标定体积尺度',85)
            recognition('board-scale','--bundle',bundle)
        gs('fuse','--bundle',bundle,'--segmentation',seg,'--max-views',self.settings['fusion_max_views'],'--width',self.settings['fusion_width'])
        self.stage(meal_id,'检查三维关联效果',95)
        gs('evaluate','--bundle',bundle,'--output',work/'evaluation')
        report = read_json(work/'evaluation/report.json')
        if report['mean_instance_iou'] < self.settings['minimum_instance_iou'] or report['food_click_accuracy'] < self.settings['minimum_click_accuracy']:
            raise RuntimeError('三维关联效果不足，已保留二维识别清单供记录')
        self.stage(meal_id,'计算定位板标定后的体积',97)
        gs('volume','--bundle',bundle)
        def complete(value):
            if value['saved_at'] and not self.store.profile()['retain_media']: return
            value.update(bundle_ref=str(bundle),analysis={'status':'ready','stage':'三维餐盘已完成',
                 'progress':100,'error':None}, three_d_reason=None,
                 quality={'instance_iou':report['mean_instance_iou'],'click_accuracy':report['food_click_accuracy']})
        self.mutate(meal_id,complete)

    def interrupt(self, meal_id):
        """Stop an active operation, keeping its input and completed results."""
        with self._lock:
            meal = self.store.get(meal_id)
            updates = {}
            for key in ('analysis','three_d_analysis'):
                status = meal.get(key, {})
                if status.get('status') in {'queued','running'}:
                    updates[key] = {**status, 'status':'cancelled', 'stage':'识别与重建已中断' if key=='analysis' else '三维关联已中断', 'error':None}
            # A photo can finish recognition just before it queues 3D work.
            # Signal that operation too, without changing completed results.
            if not updates and not self._pending.get(meal_id): return meal
            self._signal_stop(meal_id)
            if updates: self.store.fields(meal_id, **updates)
            process = self.processes.get(meal_id)
        self._terminate(process)
        return self.store.get(meal_id)

    def _signal_stop(self, meal_id):
        self.cancelled.add(meal_id)
        event = self._events.get(meal_id)
        if event: event.set()
        for future in list(self._pending.get(meal_id, ())): future.cancel()

    @staticmethod
    def _terminate(process):
        if process and process.poll() is None:
            terminate_worker(process)
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)

    def cancel(self, meal_id):
        with self._lock:
            futures = list(self._pending.get(meal_id, ()))
            self._signal_stop(meal_id)
            process = self.processes.get(meal_id)
        self._terminate(process)
        # Asset deletion must wait for the worker to finish writing its log/results.
        for future in futures:
            if not future.cancelled():
                try: future.result(timeout=15)
                except CancelledError: pass

    def close(self):
        for meal_id in list(self.futures): self.cancel(meal_id)
        self.pool.shutdown(wait=True, cancel_futures=True)

def read_yaml(path):
    return yaml.safe_load(path.read_text(encoding='utf-8'))
