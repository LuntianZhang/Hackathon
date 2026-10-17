"""Publish an already-computed LLM run as a new app draft and verify fixed geometry."""
import argparse
from pathlib import Path
import shutil
import threading

from food3d.io import read_json, write_json
from food3d.paths import ProjectPaths
from pansense.assistance_3d import source_binding, build_selected_3d
from pansense.jobs import Jobs
from pansense.nutrition import Catalog
from pansense.store import Store


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',required=True)
    parser.add_argument('--photo-result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    project=ProjectPaths().project
    store=Store(ProjectPaths().data_root/'pansense',Catalog(project/'app/catalog/foods.json'))
    parent=store.get(args.parent)
    binding=source_binding(parent,project,store.root/'uploads')
    if not binding: raise ValueError('Parent has no verified camera binding')
    result=read_json(args.photo_result/'recognition.json')
    meal=store.create('photo',title='大模型多视图识别 · 验证',assistance={'mode':'llm','foods':[],'instances':[]},
        source_sha256=binding['source_sha256'],reconstruction_binding=binding,semantic_auto_3d=True,
        recognition_parent_id=parent['id'])
    destination=store.root/'uploads'/meal['id']/'llm_photo'
    shutil.copytree(args.photo_result,destination)
    result['photo']=str((destination/'photo.jpg').resolve())
    result['overlay']=str((destination/'overlay.png').resolve())
    result['segmentation_path']=str((destination/'multiview').resolve())
    entries=[]
    for item in result['items']:
        item['mask_path']=str((destination/f"mask_{item['instance_id']}.png").resolve())
        entries.append(store.new_item(item['name'],item['catalog_key'],item['instance_id'],
            source='automatic',name_source='vision_llm',object_kind=item['kind'],needs_review=True,
            region={'kind':'mask','path':item['mask_path']},recognition_score=item['recognition_score'],
            recognition_margin=None))
    write_json(destination/'recognition.json',result)
    snapshot=store.fields(meal['id'],items=entries,source_photo=result['photo'],input_file=result['photo'],
        recognition_result=str(destination/'recognition.json'),
        semantic_summary={'mode':'llm','instance_count':len(entries),'recognition_views':len(result['keyframes']),
                          'model':result['model'],'elapsed_seconds':result['elapsed_seconds']},
        analysis={'status':'ready','stage':'大模型多视图识别完成','progress':100,'error':None})
    jobs=Jobs(store,project,threading.Lock(),enabled=False)
    try: build_selected_3d(jobs,meal['id'],binding,snapshot)
    finally: jobs.close()
    value=store.get(meal['id'])
    write_json(args.output,{'meal_id':value['id'],'bundle':value.get('bundle_ref'),
        'quality':value.get('quality'),'three_d_analysis':value.get('three_d_analysis'),
        'recognition_result':str(destination/'recognition.json')})
    if not value.get('bundle_ref'): raise RuntimeError(value['three_d_analysis'])
    print(value['id'],value['quality'])


if __name__=='__main__': main()
