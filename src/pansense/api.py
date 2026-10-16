from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from hashlib import sha256
import json
from pathlib import Path
import shutil
import threading
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from food3d.io import read_json
from food3d.paths import ProjectPaths
from food3d.schemas import load_scene
from .jobs import Jobs
from .nutrition import Catalog
from .rendering import RenderBusy, Views
from .store import ConflictError, Store

class Body(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class ProfilePatch(Body):
    onboarded: bool | None = None
    display_name: str | None = Field(None, min_length=1,max_length=40)
    goal_kcal: float | None = Field(None,ge=100,le=10000)
    timezone: str | None = None
    retain_media: bool | None = None

class Revision(Body):
    revision: int = Field(ge=1)

class VideoStart(Revision):
    assistance: dict
    paper_present: bool = True

class Region(Body):
    kind: str = 'bbox'
    bbox: list[float] = Field(min_length=4,max_length=4)

class FoodEdit(Revision):
    name: str = Field(min_length=1,max_length=80)
    catalog_key: str | None = None
    region: Region | None = None

class MealSave(Revision):
    title: str = Field(min_length=1,max_length=80)
    meal_type: str
    occurred_at: str
    note: str = Field('',max_length=1000)

class DemoBody(Body):
    scene: str = 'fruit2'

class ViewBody(Body):
    frame_id: str | None = None
    base_token: str | None = None
    yaw: float = Field(0,ge=-360,le=360)
    pitch: float = Field(0,ge=-75,le=75)
    zoom: float = Field(1,ge=.5,le=2.5)
    food_id: str | None = None

class PickBody(Body):
    token: str
    x: float = Field(ge=0,le=1)
    y: float = Field(ge=0,le=1)

class PhotoPickBody(Body):
    x: float = Field(ge=0,le=1)
    y: float = Field(ge=0,le=1)

class DeleteBody(Body):
    confirmation: str

def create_app(project: Path | None = None, data_dir: Path | None = None, jobs_enabled=True):
    project = (project or Path(__file__).resolve().parents[2]).resolve()
    paths = ProjectPaths(project)
    data_root = paths.data_root
    root = paths.resolve(data_dir) if data_dir is not None else data_root/'pansense'
    catalog = Catalog(project/'app/catalog/foods.json')
    store = Store(root, catalog)
    gpu_lock = threading.Lock()
    views = Views(gpu_lock)
    jobs = Jobs(store, project, gpu_lock, enabled=jobs_enabled, release_gpu=views.release_gpu)
    from .scans import Scans
    scans = Scans(store,jobs,project)

    @asynccontextmanager
    async def lifespan(app):
        jobs.recover()
        yield
        jobs.close()
        scans.gateway.close()
        views.release_gpu()

    app = FastAPI(title='PanSense P0',version='0.1.0',lifespan=lifespan)
    app.state.store, app.state.jobs, app.state.views, app.state.scans = store, jobs, views, scans
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1','localhost','testserver'])

    @app.middleware('http')
    async def privacy_headers(request: Request, call_next):
        if request.method not in {'GET','HEAD','OPTIONS'}:
            origin = request.headers.get('origin')
            if origin and urlsplit(origin).netloc != request.headers.get('host'):
                return JSONResponse({'detail':'不接受来自其他网站的写入请求'}, status_code=403)
        response = await call_next(request)
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        if request.url.path.startswith('/api/'):
            response.headers['Cache-Control'] = 'no-store'
        elif request.url.path == '/' or Path(request.url.path).suffix in {'.html', '.js', '.css', '.webmanifest'}:
            # Revalidate the shell so an updated UI never mixes old scripts and styles.
            response.headers['Cache-Control'] = 'no-cache'
        return response

    @app.exception_handler(KeyError)
    async def not_found(request, error):
        return JSONResponse({'detail':str(error.args[0])}, status_code=404)
    @app.exception_handler(ConflictError)
    async def conflict(request, error):
        return JSONResponse({'detail':str(error)}, status_code=409)
    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({'detail':str(error)}, status_code=400)
    @app.exception_handler(RenderBusy)
    async def busy(request, error):
        return JSONResponse({'detail':str(error)}, status_code=409)
    @app.exception_handler(RequestValidationError)
    async def invalid_body(request, error):
        return JSONResponse({'detail':'输入格式有误，请检查填写内容',
           'fields':[{'field':'.'.join(map(str,item['loc'])),'message':item['msg']} for item in error.errors()]},status_code=422)

    def public(meal):
        value = dict(meal)
        value['model_url'] = f'/model.html?run={meal["id"]}' if meal.get('bundle_ref') else None
        has_masks = bool(meal.get('recognition_result')) or any((item.get('region') or {}).get('kind')=='mask' for item in meal['items'])
        value['semantic_overlay_url'] = f'/api/meals/{meal["id"]}/semantics' if has_masks and meal.get('source_photo') else None
        value['semantic_summary'] = meal.get('semantic_summary') or {
            'mode': meal.get('assistance',{}).get('mode','auto'),
            'instance_count':len([item for item in meal['items'] if item.get('instance_id')])}
        from .assistance_3d import source_binding
        binding = source_binding(meal, project, root/'uploads')
        value['can_refine_semantics'] = bool(binding) or meal['source_kind']=='photo' and bool(meal.get('input_file'))
        value['reconstruction_available'] = binding is not None
        value['semantic_targets_3d'] = binding is not None
        value['photo_url'] = f'/api/meals/{meal["id"]}/photo' if meal.get('source_photo') else None
        value['three_d_available'] = bool(meal.get('bundle_ref'))
        value['video_reference_ready'] = bool(meal.get('prepared_video')) and not meal.get('video_started')
        value.pop('prepared_video',None)
        from food3d.volume import published_measurements
        measurements=published_measurements(Path(meal['bundle_ref'])) if meal.get('bundle_ref') else {'status':'unavailable','instances':[]}
        value['volume_summary']={k:v for k,v in measurements.items() if k!='instances'}
        value['volume_summary']['reference_image_url']=f'/api/meals/{meal["id"]}/scale-reference' if meal.get('bundle_ref') and (Path(meal['bundle_ref'])/'measurements/paper_detections.jpg').is_file() else None
        by_instance={entry['instance_id']:entry for entry in measurements['instances']}
        value.pop('source_photo',None)
        value.pop('bundle_ref',None)
        value.pop('input_file',None)
        value.pop('recognition_result',None)
        value.pop('reconstruction_binding',None)
        value['items'] = [{**{key:val for key,val in item.items() if key != 'region'},
                         'volume':by_instance.get(item.get('instance_id')),
                         'region_available':bool(item.get('region')),
                         'region_url':f'/api/meals/{meal["id"]}/foods/{item["id"]}/region' if item.get('region') else None}
                         for item in meal['items']]
        return value

    def owned_path(path):
        resolved = Path(path).resolve()
        if not resolved.is_relative_to((root/'uploads').resolve()):
            raise ValueError('不允许访问其他目录中的上传数据')
        return resolved

    def image_path(meal):
        if not meal.get('source_photo'): raise KeyError('没有保存这餐的照片')
        path = Path(meal['source_photo']).resolve()
        allowed = [root/'uploads',data_root/'scenes',data_root/'outputs']
        if not any(path.is_relative_to(parent.resolve()) for parent in allowed) or not path.is_file():
            raise KeyError('照片已不可用')
        return path

    def clean_uploads(identifier):
        path = owned_path(root/'uploads'/identifier)
        if path.is_dir(): shutil.rmtree(path)

    @app.get('/api/health')
    def health():
        return {'status':'ok','service':'PanSense P0','storage':'local','catalog_foods':len(catalog.foods)}

    @app.get('/api/profile')
    def profile(): return store.profile()
    @app.patch('/api/profile')
    def profile_edit(body: ProfilePatch):
        patch = body.model_dump(exclude_unset=True)
        if any(value is None for key,value in patch.items() if key != 'goal_kcal'):
            raise ValueError('偏好字段不能为空')
        return store.update_profile(patch)

    @app.get('/api/catalog')
    def foods(): return {'foods':list(catalog.foods.values()),'source':catalog.data['source'],'basis':catalog.data['basis']}

    @app.get('/api/meals')
    def list_meals(date: str | None = None, drafts=False):
        return {'meals':[public(meal) for meal in store.list(date,include_drafts=drafts)
                         if meal['source_kind'] != 'assist-session' and (not meal.get('comparison_parent') or meal.get('comparison_imported'))]}
    @app.get('/api/summary')
    def summary(date: str): return store.summary(date)
    @app.post('/api/meals/manual',status_code=201)
    def manual(): return public(store.create('manual',three_d_reason='手工记录没有三维资产，可使用食物清单'))

    @app.post('/api/meals/demo',status_code=201)
    def demo(body: DemoBody):
        bundles = {'fruit2':data_root/'outputs/fruit2_bundle','fruit1':data_root/'outputs/fruit1_quality_cleaned'}
        bundle = bundles.get(body.scene)
        if bundle is None or not (bundle/'semantics/instances.json').is_file():
            raise ValueError('该三维示例暂不可用')
        instances = read_json(bundle/'semantics/instances.json')
        _, frames = load_scene(bundle/'scene.json')
        reference = next((frame for frame in frames if frame.frame_id == instances.get('reference_frame')),frames[0])
        entries = [store.new_item(item['name'],instance_id=item['instance_id'],source='demo',needs_review=True)
                   for item in instances['instances'] if item['instance_id'] > 0]
        # Demo source masks are referenced safely and never copied over the POC.
        seg = Path(read_json(bundle/'semantics/fusion.json')['segmentation'])
        for item in entries:
            item['region'] = {'kind':'mask','path':str(seg/'masks'/f'{reference.frame_id}.png'),'instance_id':item['instance_id']}
        return public(store.create('demo',title='水果餐盘 · 三维示例', source_photo=str((bundle/reference.image_path).resolve()),
                     bundle_ref=str(bundle.resolve()),items=entries,
                     analysis={'status':'ready','stage':'已有三维示例','progress':100,'error':None}))

    @app.post('/api/meals/upload',status_code=201)
    async def upload(file: UploadFile = File(...), kind: str = 'photo', assistance: str = Form(''),
                     prepare: bool = Form(False), paper_present: bool = Form(True)):
        if kind not in {'photo','video'}: raise ValueError('请选择照片或视频')
        if prepare and kind != 'video': raise ValueError('仅环绕视频需要准备参考帧')
        from food3d.segmentation.hints import validate_assistance
        try:
            options = validate_assistance(json.loads(assistance) if assistance else None, list(catalog.foods.values()), allow_none=True)
        except json.JSONDecodeError as error:
            raise ValueError('辅助识别配置不是有效 JSON') from error
        if kind == 'video' and options['mode'] == 'manual':
            raise ValueError('请先准备视频参考帧，再点击或框选食物')
        meal = store.create(kind,assistance=options,paper_present=paper_present,
                            analysis={'status':'queued','stage':'接收文件','progress':0,'error':None})
        folder = owned_path(root/'uploads'/meal['id'])
        folder.mkdir(parents=True)
        temp = folder/'incoming'
        limit = (20 if kind == 'photo' else 240)*1024*1024
        count = 0
        try:
            with temp.open('wb') as stream:
                while chunk := await file.read(1024*1024):
                    count += len(chunk)
                    if count > limit: raise ValueError(f'文件过大：照片上限 20 MB，视频上限 240 MB')
                    stream.write(chunk)
            if count == 0: raise ValueError('文件为空，请重新选择')
            if kind == 'photo':
                try:
                    with Image.open(temp) as source:
                        if source.width*source.height > 40_000_000: raise ValueError('照片像素过高，请缩小后再试')
                        picture = ImageOps.exif_transpose(source).convert('RGB')
                        picture.thumbnail((960,960))
                        destination = folder/'source.jpg'
                        picture.save(destination,quality=95) # Strip EXIF location metadata.
                except (UnidentifiedImageError, OSError) as error:
                    raise ValueError('无法读取照片，请使用 JPG、PNG 或 WebP') from error
                temp.unlink()
                store.fields(meal['id'],source_photo=str(destination),source_sha256=sha256(destination.read_bytes()).hexdigest())
            else:
                destination = folder/'source.mp4'
                temp.rename(destination)
            store.fields(meal['id'],input_file=str(destination))
            jobs.submit(meal['id'],'video-prepare' if prepare else kind,destination)
            return public(store.get(meal['id']))
        except Exception:
            if store.exists(meal['id']): store.delete(meal['id'])
            clean_uploads(meal['id'])
            raise
        finally:
            await file.close()

    @app.get('/api/meals/{identifier}')
    def get_meal(identifier: str): return public(store.get(identifier))

    @app.post('/api/meals/{identifier}/video-start')
    def start_prepared_video(identifier: str, body: VideoStart):
        from food3d.segmentation.hints import validate_assistance
        options = validate_assistance(body.assistance, list(catalog.foods.values()), allow_none=True)
        def start(value):
            if value['source_kind'] not in {'video','scan'} or not value.get('prepared_video'):
                raise ValueError('请先准备视频参考帧')
            if value['analysis']['status'] != 'ready' or value.get('video_started'):
                raise ValueError('该视频尚未准备好，或已经开始分析')
            value.update(assistance=options, paper_present=body.paper_present, video_started=True,
                         analysis={'status':'queued','stage':'等待分析','progress':0,'error':None})
        value = store.mutate(identifier, start, body.revision)
        jobs.submit(identifier,'scan' if value['source_kind']=='scan' else 'video',owned_path(value['input_file']))
        return public(store.get(identifier))
    @app.post('/api/meals/{identifier}/retry')
    def retry(identifier: str):
        meal = store.get(identifier)
        if meal['analysis']['status'] in {'running','queued'}: raise ValueError('这餐正在分析，请稍候')
        if not meal.get('input_file') or not Path(meal['input_file']).exists(): raise ValueError('源文件已不可用，可继续手工记录')
        kind = meal['source_kind']+'-prepare' if meal['source_kind'] in {'video','scan'} and meal.get('video_preparing') else meal['source_kind']
        jobs.submit(identifier,kind,owned_path(meal['input_file']))
        return public(store.get(identifier))

    @app.post('/api/meals/{identifier}/foods',status_code=201)
    def add_food(identifier: str, body: FoodEdit):
        region = body.region.model_dump() if body.region else None
        if region:
            box = region['bbox']
            if region['kind'] != 'bbox' or not all(0<=number<=1 for number in box) or not (box[0]<box[2] and box[1]<box[3]):
                raise ValueError('食物区域必须位于照片内且具有正面积')
        return public(store.add_item(identifier,body.name,body.catalog_key,body.revision,region))
    @app.patch('/api/meals/{identifier}/foods/{food_id}')
    def edit_food(identifier: str,food_id: str,body: FoodEdit):
        return public(store.edit_item(identifier,food_id,body.name,body.catalog_key,body.revision))
    @app.delete('/api/meals/{identifier}/foods/{food_id}')
    def delete_food(identifier: str,food_id: str,revision: int):
        return public(store.delete_item(identifier,food_id,revision))

    @app.post('/api/meals/{identifier}/save')
    def save(identifier: str,body: MealSave):
        if body.meal_type not in {'breakfast','lunch','dinner','snack'}: raise ValueError('请选择早餐、午餐、晚餐或加餐')
        value = store.save(identifier,body.model_dump(exclude={'revision'}),body.revision)
        if not store.profile()['retain_media']:
            jobs.cancel(identifier)
            def strip(meal):
                meal.update(source_photo=None,bundle_ref=None,input_file=None,three_d_reason='按隐私偏好仅保存食物记录',
                            prepared_video=None,video_preparing=False,
                            recognition_result=None,semantic_summary=None,reconstruction_binding=None,semantic_auto_3d=False,
                            analysis={'status':'ready','stage':'仅保存文字与营养记录','progress':100,'error':None})
                for item in meal['items']: item['region'] = None
            value = store.mutate(identifier,strip)
            views.clear_meal(identifier)
            clean_uploads(identifier)
            scans.remove_for_meal(identifier)
        return public(value)

    @app.delete('/api/meals/{identifier}')
    def delete_meal(identifier: str):
        store.delete(identifier)
        jobs.cancel(identifier)
        scans.remove_for_meal(identifier)
        views.clear_meal(identifier)
        clean_uploads(identifier)
        return {'deleted':True}

    @app.get('/api/meals/{identifier}/photo')
    def photo(identifier: str): return FileResponse(image_path(store.get(identifier)))
    @app.get('/api/meals/{identifier}/foods/{food_id}/region')
    def region(identifier: str,food_id: str):
        meal = store.get(identifier)
        item = next((item for item in meal['items'] if item['id']==food_id),None)
        if not item or not item.get('region'): raise KeyError('这项食物没有视觉区域')
        info = item['region']
        source = Image.open(image_path(meal))
        if info['kind'] == 'mask':
            path = Path(info['path']).resolve()
            if not any(path.is_relative_to(parent.resolve()) for parent in [root/'uploads',data_root/'outputs']): raise KeyError('食物区域不可用')
            labels = np_array(Image.open(path))
            mask = (labels==info['instance_id']) if info.get('instance_id') else labels>0
            alpha = Image.fromarray((mask.astype('uint8')*130)).resize(source.size)
        else:
            from PIL import ImageDraw
            alpha = Image.new('L',source.size)
            x0,y0,x1,y1 = info['bbox']
            ImageDraw.Draw(alpha).rectangle([x0*source.width,y0*source.height,x1*source.width,y1*source.height],fill=130)
        overlay = Image.new('RGBA',source.size,(30,180,150,0)); overlay.putalpha(alpha)
        stream = BytesIO(); overlay.save(stream,format='PNG')
        return Response(stream.getvalue(),media_type='image/png')

    @app.post('/api/meals/{identifier}/view')
    def view(identifier: str,body: ViewBody):
        return views.render(store.get(identifier),**body.model_dump())
    @app.post('/api/meals/{identifier}/pick')
    def pick(identifier: str,body: PickBody):
        return views.pick(store.get(identifier),body.token,body.x,body.y)
    @app.post('/api/meals/{identifier}/photo-pick')
    def photo_pick(identifier: str,body: PhotoPickBody):
        meal = store.get(identifier)
        image_path(meal)
        for item in reversed(meal['items']):
            info = item.get('region')
            if not info: continue
            if info['kind'] == 'bbox':
                x0,y0,x1,y1 = info['bbox']
                inside = x0<=body.x<=x1 and y0<=body.y<=y1
            else:
                path = Path(info['path']).resolve()
                if not any(path.is_relative_to(parent.resolve()) for parent in [root/'uploads',data_root/'outputs']): continue
                if not path.is_file(): continue
                labels = np_array(Image.open(path))
                pixel = labels[min(labels.shape[0]-1,int(body.y*labels.shape[0])),min(labels.shape[1]-1,int(body.x*labels.shape[1]))]
                inside = pixel == info['instance_id'] if info.get('instance_id') else pixel > 0
            if inside: return {'food_id':item['id'],'message':item['name']}
        return {'food_id':None,'message':'背景或尚未关联的食物区域'}
    @app.get('/api/meals/{identifier}/views/{token}/image')
    def rendered(identifier: str,token: str):
        store.get(identifier)
        return Response(views.entry(token,identifier)['image'],media_type='image/jpeg')

    @app.get('/api/privacy/export')
    def export():
        data = {'schema_version':1,'profile':store.profile(),'meals':[public(meal) for meal in store.list(include_drafts=True)],
                'note':'营养记录与偏好导出；照片和三维模型另存本机，未包含在此 JSON 中'}
        return Response(json.dumps(data,ensure_ascii=False,indent=2),media_type='application/json',
                        headers={'Content-Disposition':'attachment; filename="PanSense-records.json"'})
    @app.post('/api/privacy/delete')
    def purge(body: DeleteBody):
        if body.confirmation != 'DELETE': raise ValueError('请确认删除本机全部餐食记录')
        identifiers = [meal['id'] for meal in store.list(include_drafts=True)]
        store.purge()
        scans.purge()
        for identifier in identifiers:
            jobs.cancel(identifier); clean_uploads(identifier); views.clear_meal(identifier)
        return {'deleted':True,'profile':store.profile()}

    from .assistance_api import register_assistance
    register_assistance(app, store, jobs, project, public, owned_path)
    from .meal_semantics import register_meal_semantics
    register_meal_semantics(app, store, jobs, project, public, owned_path)
    from .scan_api import register_scan_routes
    register_scan_routes(app,scans,public)
    app.mount('/',StaticFiles(directory=project/'app/web',html=True),name='pansense-web')
    return app

def np_array(image):
    import numpy as np
    return np.asarray(image)
