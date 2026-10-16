"""Three independent recognition variants on the exact same source photo."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import shutil

from fastapi import File, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

from food3d.io import read_json
from food3d.paths import ProjectPaths
from food3d.segmentation.hints import validate_assistance
from .assistance_3d import source_binding


def register_assistance(app, store, jobs, project, public, owned_path):
    data_root = ProjectPaths(project).data_root
    def session(identifier):
        meal = store.get(identifier)
        if meal['source_kind'] != 'assist-session':
            raise ValueError('请选择辅助识别对比照片')
        return meal

    @app.get('/api/assist/sessions')
    def sessions():
        return {'sessions': [public(m) for m in store.list(include_drafts=True) if m['source_kind'] == 'assist-session']}

    @app.post('/api/assist/sessions', status_code=201)
    async def create_session(file: UploadFile | None = File(None)):
        meal = store.create('assist-session', title='辅助识别对比', analysis={
            'status':'ready', 'stage':'选择一种方式开始比较', 'progress':100, 'error':None})
        folder = owned_path(store.root/'uploads'/meal['id'])
        folder.mkdir(parents=True)
        temporary, photo = folder/'incoming', folder/'source.jpg'
        try:
            if file:
                count = 0
                with temporary.open('wb') as stream:
                    while chunk := await file.read(1024*1024):
                        count += len(chunk)
                        if count > 20*1024*1024:
                            raise ValueError('照片请小于 20 MB')
                        stream.write(chunk)
                source = temporary
            else:
                source = data_root/'outputs/fruit2_view_counts_run01/n30/independent_colmap_scene/images/000018.jpg'
                if not source.is_file():
                    raise ValueError('示例照片暂不可用，请上传照片')
            with Image.open(source) as image:
                if image.width*image.height > 40_000_000:
                    raise ValueError('照片像素过高，请缩小后再试')
                picture = ImageOps.exif_transpose(image).convert('RGB')
                picture.thumbnail((960,960))
                picture.save(photo, quality=95)
            digest = sha256(photo.read_bytes()).hexdigest()
            return public(store.fields(meal['id'], source_photo=str(photo), input_file=str(photo), source_sha256=digest))
        except (UnidentifiedImageError, OSError) as error:
            store.delete(meal['id'])
            shutil.rmtree(folder)
            raise ValueError('无法读取照片，请使用 JPG、PNG 或 WebP') from error
        except Exception:
            store.delete(meal['id'])
            shutil.rmtree(folder)
            raise
        finally:
            temporary.unlink(missing_ok=True)
            if file:
                await file.close()

    @app.get('/api/assist/sessions/{identifier}/runs')
    def runs(identifier: str):
        session(identifier)
        return {'runs':[public(m) for m in store.list(include_drafts=True) if m.get('comparison_parent') == identifier]}

    @app.post('/api/assist/sessions/{identifier}/runs', status_code=201)
    def run(identifier: str, body: dict):
        parent = session(identifier)
        assistance = validate_assistance(body, list(store.catalog.foods.values()))
        if not parent.get('input_file') or not Path(parent['input_file']).is_file():
            raise ValueError('原照片已删除，请重新上传')
        children = [m for m in store.list(include_drafts=True) if m.get('comparison_parent') == identifier]
        if len(children) >= 40:
            raise ValueError('每张照片最多保留 40 次对比，请删除旧结果后重试')
        # Disable duplicates on double-click while still permitting three modes
        # to queue for one source. No result can overwrite a sibling or 3D model.
        if any(m['analysis']['status'] in {'queued','running'} and m.get('assistance') == assistance for m in children):
            raise ValueError('相同输入正在分析，请等待结果')
        meal = store.create('photo', title='辅助识别 · '+assistance['mode'], comparison_parent=identifier,
                            source_sha256=parent['source_sha256'], assistance=assistance)
        folder = owned_path(store.root/'uploads'/meal['id'])
        folder.mkdir(parents=True)
        photo = folder/'source.jpg'
        shutil.copyfile(owned_path(parent['input_file']), photo)
        store.fields(meal['id'], input_file=str(photo), source_photo=str(photo))
        jobs.submit(meal['id'], 'photo', photo)
        return public(store.get(meal['id']))

    def result(identifier):
        meal = store.get(identifier)
        if not meal.get('recognition_result'):
            raise ValueError('分割结果尚未生成')
        path = owned_path(meal['recognition_result'])
        return meal, read_json(path), path.parent

    @app.get('/api/assist/runs/{identifier}/result')
    def get_result(identifier: str):
        meal, value, _ = result(identifier)
        return {**{k:v for k,v in value.items() if k not in {'photo','overlay','items'}},
            'source_sha256': meal.get('source_sha256'),
            'overlay_url': f'/api/assist/runs/{identifier}/overlay',
            'items':[{k:v for k,v in item.items() if k != 'mask_path'} for item in value['items']]}

    @app.get('/api/assist/runs/{identifier}/overlay')
    def overlay(identifier: str):
        _, _, folder = result(identifier)
        return FileResponse(folder/'overlay.png')

    @app.post('/api/assist/sessions/{identifier}/choice')
    def choose(identifier: str, body: dict):
        session(identifier)
        if set(body) != {'run_id'} or not isinstance(body['run_id'], str):
            raise ValueError('请选择一个已完成的对比版本')
        selected = store.get(body['run_id'])
        if selected.get('comparison_parent') != identifier or selected['analysis']['status'] != 'ready':
            raise ValueError('只能选择这张照片的已完成结果')
        value = store.fields(identifier, preferred_run=selected['id'], preferred_mode=selected['assistance']['mode'])
        prepare_3d(selected, value)
        return public(store.get(identifier))

    def prepare_3d(selected, parent):
        binding = source_binding(parent, project, store.root/'uploads')
        if not binding:
            store.fields(selected['id'], three_d_reason='这张单独上传的照片没有关联的多视角重建数据；请使用已有模型参考照片，或先导入环绕视频完成重建')
            return
        if not selected['items']:
            store.fields(selected['id'], three_d_reason='这个版本没有食物实例，请调整识别结果后再生成三维关联')
            return
        jobs.submit_3d(selected['id'], binding)

    @app.post('/api/assist/runs/{identifier}/three-d')
    def start_three_d(identifier: str):
        selected = store.get(identifier)
        if not selected.get('comparison_parent') or selected['analysis']['status'] != 'ready':
            raise ValueError('请选择已完成的识别版本')
        prepare_3d(selected, session(selected['comparison_parent']))
        return public(store.get(identifier))
