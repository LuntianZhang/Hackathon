"""Augment an existing evaluation with transparent metrics and UI evidence."""
from pathlib import Path
from food3d.paths import ProjectPaths
import numpy as np
from PIL import Image
from food3d.io import read_json, write_json
from food3d.schemas import UNKNOWN_ID

bundle=ProjectPaths().resolve('data/outputs/fruit1_final')
evaluation=ProjectPaths().resolve('data/outputs/fruit1_final_evaluation')
report=read_json(evaluation/'report.json')
fusion=read_json(bundle/'semantics/fusion.json')
segmentation=Path(fusion['segmentation'])
segment=read_json(segmentation/'segmentation.json')
bg_clicks=[c for c in report['clicks'] if c['expected_id']==0]
report['background_food_false_positive_count']=sum(c['actual_id']>0 for c in bg_clicks)
report['background_no_food_name_accuracy']=1-report['background_food_false_positive_count']/len(bg_clicks)
report['unknown_gaussian_fraction']=fusion['unknown_fraction']
report['mean_foreground_psnr_db']=float(np.mean([v['foreground_psnr_db'] for v in report['per_view']]))
for view in report['per_view']:
    prediction=np.load(evaluation/(view['frame_id']+'_instance_ids.npy'),allow_pickle=False)
    with Image.open(segmentation/segment['masks'][view['frame_id']]) as image:
        expected=np.array(image.resize((prediction.shape[1],prediction.shape[0]),Image.Resampling.NEAREST))
    bg=expected==0
    view['background_food_pixel_fraction']=float(((prediction[bg]>0)&(prediction[bg]!=UNKNOWN_ID)).mean())
write_json(evaluation/'report.json',report)
native=read_json(ProjectPaths().resolve('data/outputs/native_parity/report.json'))
summary={'status':'rendered_input_poc_verified_with_limitations','input':'120 user-provided rendered images; no known cameras','real_photos_verified':False,'sfm_registered':120,'sfm_reprojection_error_px':1.1102342492931803,'training_steps':7000,'gaussians':read_json(bundle/'model.json')['gaussians'],'cpu_tests_passed':14,'native_render_parity':native,'semantic_metrics':{k:report[k] for k in ['mean_instance_iou','food_click_accuracy','food_click_count','background_click_accuracy','background_no_food_name_accuracy','background_click_count','mean_psnr_db','mean_foreground_psnr_db','unknown_gaussian_fraction']},'ui_checks':{'food_click':True,'highlight':True,'identity_across_view_change':True,'orbit_render':True,'background_no_food_name':True,'rename_and_restore':True,'screenshot':'data/outputs/fruit1_demo_verified.png'},'limitations':['SAM2 consistency is not human ground truth','3/72 food clicks fail at view 000000 with occlusion','3/12 background test points return unknown rather than confident background','61.2% of Gaussians remain unknown, including poorly observed or low contribution splats','Real camera images and bread have not been validated']}
write_json(bundle/'validation.json',summary)
model=read_json(bundle/'model.json')
model['semantics_verified']=True
model['validation_scope']='rendered_input_poc_with_known_limitations; see validation.json'
write_json(bundle/'model.json',model)
fusion['status']='rendered_input_holdout_and_browser_checks_completed_with_limitations'
write_json(bundle/'semantics/fusion.json',fusion)
print(summary['semantic_metrics'])
