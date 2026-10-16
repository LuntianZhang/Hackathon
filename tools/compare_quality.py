"""Compare identical held-out views; include apple color coverage diagnostics."""
import argparse
from pathlib import Path
from food3d.paths import ProjectPaths, resolve_arguments
import numpy as np
from PIL import Image
from food3d.io import read_json, write_json
from food3d.schemas import load_scene, UNKNOWN_ID
from food3d.visuals import contact_sheet


def appearance_metrics(bundle: Path, evaluation: Path) -> dict:
    report=read_json(evaluation/'report.json')
    fusion=read_json(bundle/'semantics/fusion.json')
    segment_root=Path(fusion['segmentation'])
    segment=read_json(segment_root/'segmentation.json')
    _,frames=load_scene(bundle/'scene.json')
    frames={frame.frame_id:frame for frame in frames}
    apple_ids=[entry['instance_id'] for entry in read_json(bundle/'semantics/instances.json')['instances'] if entry.get('auto_label')=='apple']
    squared_error={i:0. for i in range(1,7)}
    pixel_counts={i:0 for i in squared_error}
    source_red=recovered_red=unknown_pixels=food_pixels=0
    for frame_id in report['heldout_views']:
        rendered=np.array(Image.open(evaluation/(frame_id+'_render.png')).convert('RGB'),dtype=float)/255
        size=(rendered.shape[1],rendered.shape[0])
        frame=frames[frame_id]
        with Image.open(bundle/frame.image_path) as image:
            source=np.array(image.convert('RGB').resize(size,Image.Resampling.LANCZOS),dtype=float)/255
        with Image.open(segment_root/segment['masks'][frame_id]) as image:
            labels=np.array(image.resize(size,Image.Resampling.NEAREST))
        predicted=np.load(evaluation/(frame_id+'_instance_ids.npy'),allow_pickle=False)
        error=(source-rendered)**2
        for instance_id in squared_error:
            region=labels==instance_id
            squared_error[instance_id]+=float(error[region].sum())
            pixel_counts[instance_id]+=int(region.sum())*3
        apples=np.isin(labels,apple_ids)
        red=apples & (source[:,:,0]>0.4) & (source[:,:,0]>1.3*source[:,:,1]) & (source[:,:,0]>1.3*source[:,:,2])
        recovered=(rendered[:,:,0]>0.4) & (rendered[:,:,0]>1.3*rendered[:,:,1]) & (rendered[:,:,0]>1.3*rendered[:,:,2])
        source_red+=int(red.sum())
        recovered_red+=int((red & recovered).sum())
        food=(labels>0)&(labels!=UNKNOWN_ID)
        food_pixels+=int(food.sum())
        unknown_pixels+=int(((predicted==UNKNOWN_ID)&food).sum())
    return {'instance_rgb_psnr_db':{str(i):float(-10*np.log10(max(squared_error[i]/max(pixel_counts[i],1),1e-12))) for i in squared_error},'apple_red_pixel_recall':recovered_red/max(source_red,1),'foreground_unknown_pixel_fraction':unknown_pixels/max(food_pixels,1),'red_coverage_limitation':'Color diagnostic on SAM2 apple masks, not a geometric completeness ground truth'}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline-bundle',type=Path,default=ProjectPaths().resolve('data/outputs/fruit1_final'))
    p.add_argument('--baseline-report',type=Path,default=ProjectPaths().resolve('data/outputs/fruit1_final_evaluation'))
    p.add_argument('--candidate-bundle',type=Path,required=True)
    p.add_argument('--candidate-report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=resolve_arguments(p.parse_args())
    baseline=read_json(a.baseline_report/'report.json')
    candidate=read_json(a.candidate_report/'report.json')
    if baseline['heldout_views']!=candidate['heldout_views']:
        raise ValueError('Comparison requires the same held-out views')
    keys=['mean_instance_iou','food_click_accuracy','background_click_accuracy','background_no_food_name_accuracy','mean_psnr_db','mean_foreground_psnr_db','unknown_gaussian_fraction']
    result={'baseline':{k:baseline[k] for k in keys},'candidate':{k:candidate[k] for k in keys},'baseline_appearance':appearance_metrics(a.baseline_bundle,a.baseline_report),'candidate_appearance':appearance_metrics(a.candidate_bundle,a.candidate_report)}
    result['delta']={k:result['candidate'][k]-result['baseline'][k] for k in keys}
    result['same_holdout_views']=baseline['heldout_views']
    a.output.mkdir(parents=True,exist_ok=True)
    write_json(a.output/'comparison.json',result)
    _,frames=load_scene(a.baseline_bundle/'scene.json')
    frames={f.frame_id:f for f in frames}
    previews=[]
    for frame_id in ['000000','000030','000080']:
        paths=[('source',a.baseline_bundle/frames[frame_id].image_path),('baseline',a.baseline_report/(frame_id+'_render.png')),('candidate',a.candidate_report/(frame_id+'_render.png'))]
        for label,path in paths:
            with Image.open(path) as image:
                picture=image.convert('RGB')
                picture.thumbnail((248,188))
                previews.append((frame_id+' '+label,picture.copy()))
    contact_sheet(previews,a.output/'comparison_sheet.png',columns=3)
    print(result)


if __name__=='__main__':
    main()
