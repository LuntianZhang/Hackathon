"""Semantic recognition inside the meal workflow, using the same saved masks."""
from pathlib import Path

from fastapi.responses import FileResponse
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field
from PIL import Image, ImageDraw
import numpy as np
from io import BytesIO

import shutil

from food3d.segmentation.hints import validate_assistance
from food3d.paths import ProjectPaths
from .store import ConflictError
from .assistance_3d import source_binding, normalized_photo


class RecognitionImport(BaseModel):
    model_config = ConfigDict(extra='forbid')
    run_id: str = Field(min_length=1, max_length=80)


class RecognitionRestart(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: int = Field(ge=1)
    assistance: dict


def register_meal_semantics(app, store, jobs, project, public, owned_path):
    data_root = ProjectPaths(project).data_root
    @app.get('/api/meals/{identifier}/scale-reference')
    def scale_reference(identifier:str):
        meal=store.get(identifier)
        if not meal.get('bundle_ref'): raise KeyError('这餐尚无纸片标定结果')
        path=(Path(meal['bundle_ref'])/'measurements/paper_detections.jpg').resolve()
        if not path.is_relative_to((data_root).resolve()) or not path.is_file(): raise KeyError('纸片检测图片不可用')
        return FileResponse(path)

    @app.get('/api/recognition/results')
    def recognition_results():
        return {'results':[public(m) for m in store.list(include_drafts=True)
            if m.get('comparison_parent') and m['analysis']['status']=='ready' and m['items']][:20]}

    @app.get('/api/recognition/reference-photo')
    def reference_photo():
        path = data_root/'outputs/fruit2_view_counts_run01/n30/independent_colmap_scene/images/000018.jpg'
        if not path.is_file(): raise KeyError('参考照片暂不可用，请上传自己的照片')
        return FileResponse(path)

    @app.post('/api/meals/{identifier}/recognize')
    def recognize_again(identifier: str, body: RecognitionRestart):
        original = store.get(identifier)
        if original['revision'] != body.revision:
            raise ConflictError('餐食已更新，请刷新后再调整识别')
        binding = source_binding(original, project)
        if not binding and (original['source_kind'] != 'photo' or not original.get('input_file')):
            raise ValueError('这餐缺少可重新识别的原始照片，请重新导入照片')
        photo_input=original.get('input_file') if original['source_kind']=='photo' else None
        source = Path(photo_input or original.get('source_photo') or '').resolve() if binding else owned_path(original['input_file'])
        if not source.is_file(): raise ValueError('原照片已不可用，请重新导入')
        options = validate_assistance(body.assistance,list(store.catalog.foods.values()))
        # A new draft keeps the saved meal, manual corrections and its 3D
        # semantics intact until the user chooses and saves the new result.
        value = store.create('photo',title='这一餐 · 重新识别',assistance=options,
            recognition_parent_id=identifier,source_sha256=binding['source_sha256'] if binding else original.get('source_sha256'),
            reconstruction_binding=binding,semantic_auto_3d=bool(binding))
        folder = owned_path(store.root/'uploads'/value['id']);folder.mkdir(parents=True)
        try:
            destination=folder/'source.jpg'
            if binding and not photo_input: destination.write_bytes(normalized_photo(source))
            else: shutil.copyfile(source,destination)
            store.fields(value['id'],source_photo=str(destination),input_file=str(destination))
            jobs.submit(value['id'],'photo',destination)
            return public(store.get(value['id']))
        except Exception:
            jobs.cancel(value['id'])
            store.delete(value['id']);shutil.rmtree(folder)
            raise

    @app.post('/api/meals/from-recognition')
    def use_recognition(body: RecognitionImport):
        value = store.get(body.run_id)
        if not value.get('comparison_parent') or value['analysis']['status'] != 'ready' or not value['items']:
            raise ValueError('请选择已有食物实例且分析完成的识别结果')
        if value.get('comparison_imported'):
            return public(value)
        # Promote the same record: no copies, duplicate diary entries, changed
        # instance IDs, or references to another meal's deletable assets.
        return public(store.fields(value['id'], comparison_imported=True, title='这一餐 · 食物识别'))

    @app.get('/api/meals/{identifier}/semantics')
    def semantic_image(identifier: str):
        meal = store.get(identifier)
        if meal.get('recognition_result'):
            path = owned_path(meal['recognition_result']).parent/'overlay.png'
            if not path.is_file(): raise KeyError('分割图片已不可用，可继续使用食物清单')
            return FileResponse(path)
        allowed=[store.root/'uploads',data_root/'outputs',data_root/'scenes']
        def asset(value):
            path=Path(value or '').resolve()
            if not path.is_file() or not any(path.is_relative_to(root.resolve()) for root in allowed):
                raise KeyError('食物区域暂不可用')
            return path
        with Image.open(asset(meal.get('source_photo'))) as raw:
            image=raw.convert('RGB')
        pixels=np.array(image);marks=[]
        for item in meal['items']:
            region=item.get('region') or {}
            if region.get('kind')!='mask': continue
            with Image.open(asset(region.get('path'))) as raw:
                labels=np.asarray(raw.resize(image.size,Image.Resampling.NEAREST))
            mask=labels==region['instance_id'] if region.get('instance_id') else labels>0
            identifier=item.get('instance_id') or len(marks)+1
            color=np.array([(identifier*71)%220+30,(identifier*139)%220+30,(identifier*193)%220+30])
            pixels[mask]=(.55*pixels[mask]+.45*color).astype(np.uint8)
            y,x=np.nonzero(mask)
            if len(x):marks.append((identifier,int(x.min()),int(y.min())))
        if not marks: raise KeyError('这餐尚无语义分割结果')
        preview=Image.fromarray(pixels);draw=ImageDraw.Draw(preview)
        for identifier,x,y in marks:draw.text((x,y),f'ID {identifier}',fill='white',stroke_width=2,stroke_fill='black')
        stream=BytesIO();preview.save(stream,format='PNG')
        return Response(stream.getvalue(),media_type='image/png')

    @app.post('/api/meals/{identifier}/semantic-3d')
    def semantic_three_d(identifier: str):
        value = store.get(identifier)
        if value.get('bundle_ref'):
            return public(value)
        if value['analysis']['status'] != 'ready' or not value['items']:
            raise ValueError('请先完成食物识别并核对食物区域')
        binding = source_binding(value, project)
        if not binding:
            raise ValueError('这张照片尚未关联多视角重建，请先导入环绕视频，或使用已有模型的参考照片')
        jobs.submit_3d(identifier, binding)
        return public(store.get(identifier))
