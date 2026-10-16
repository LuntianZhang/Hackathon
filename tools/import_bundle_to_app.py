"""Import an existing local bundle as an unsaved meal, without GPU work."""
import argparse
import json
from pathlib import Path
import shutil

from food3d.io import read_json
from food3d.paths import ProjectPaths, resolve_arguments
from food3d.schemas import load_scene
from pansense.nutrition import Catalog
from pansense.store import Store


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--video',type=Path)
    parser.add_argument('--data-dir',type=Path,default=ProjectPaths().resolve('data/pansense_assistance'))
    parser.add_argument('--title',default='已重建场景 · 体积验证')
    args=resolve_arguments(parser.parse_args());project=Path(__file__).resolve().parents[1]
    bundle=args.bundle.resolve();root=args.data_dir.resolve()
    data_root = ProjectPaths(project).data_root
    if not bundle.is_relative_to(data_root) or not root.is_relative_to(data_root):
        raise ValueError('模型和应用数据目录必须位于项目 data 内')
    instances=read_json(bundle/'semantics/instances.json')
    fusion=read_json(bundle/'semantics/fusion.json')
    _,frames=load_scene(bundle/'scene.json')
    reference=next(f for f in frames if f.frame_id==instances['reference_frame'])
    segmentation=Path(fusion['segmentation']).resolve()
    if not segmentation.is_relative_to(data_root): raise ValueError('掩码必须位于项目 data 内')
    store=Store(root,Catalog(project/'app/catalog/foods.json'))
    items=[store.new_item(i['name'],instance_id=i['instance_id'],source='manual_prompt',needs_review=False,
           region={'kind':'mask','path':str(segmentation/'masks'/f'{reference.frame_id}.png'),'instance_id':i['instance_id']})
           for i in instances['instances'] if i['instance_id']>0]
    value=store.create('video' if args.video else 'demo',title=args.title,items=items,bundle_ref=str(bundle),
        source_photo=str((bundle/reference.image_path).resolve()),
        analysis={'status':'ready','stage':'重建及纸片体积标定已完成','progress':100,'error':None})
    if args.video:
        source=args.video.resolve()
        if not source.is_file(): store.delete(value['id']);raise FileNotFoundError(source)
        folder=(root/'uploads'/value['id']).resolve()
        if not folder.is_relative_to(root/'uploads'): raise ValueError('上传目录越界')
        folder.mkdir(parents=True)
        try:
            shutil.copy2(source,folder/'source.mp4')
            store.fields(value['id'],input_file=str(folder/'source.mp4'))
        except Exception:
            store.delete(value['id']);shutil.rmtree(folder);raise
    print(json.dumps({'meal_id':value['id'],'saved':False,'url':'http://127.0.0.1:8003/?v=paper-volume-1#plate/'+value['id']},ensure_ascii=False))


if __name__=='__main__': main()
