"""Record the reviewed quality variant and select it for the local demo."""
from pathlib import Path
from food3d.paths import ProjectPaths
from food3d.io import read_json,write_json
from food3d.reconstruction.gaussians import file_hash

bundle=ProjectPaths().resolve('data/outputs/fruit1_quality_cleaned')
report=read_json(ProjectPaths().resolve('data/outputs/fruit1_quality_cleaned_evaluation/report.json'))
comparison=read_json(ProjectPaths().resolve('data/outputs/fruit1_quality_cleaned_comparison/comparison.json'))
native=read_json(ProjectPaths().resolve('data/outputs/quality_native_parity/report.json'))
model=read_json(bundle/'model.json')
fusion=read_json(bundle/'semantics/fusion.json')
assert report['model_sha256']==model['model_sha256']==fusion['model_sha256']
assert report['food_click_accuracy']>=comparison['baseline']['food_click_accuracy']
assert report['mean_instance_iou']>comparison['baseline']['mean_instance_iou']
assert report['mean_foreground_psnr_db']>comparison['baseline']['mean_foreground_psnr_db']
assert native['passed']
assert ProjectPaths().resolve('data/outputs/fruit1_quality_demo_verified.png').is_file()
summary={'status':'rendered_input_quality_improvement_verified_with_limitations','training_steps':15000,'gaussians':model['gaussians'],'cleanup_removed_gaussians':model['postprocess']['removed_gaussians'],'test_count':17,'same_development_holdout_views':report['heldout_views'],'comparison':comparison,'native_parity_before_cleanup':native,'cleanup_changes_geometry_intentionally':True,'ply_sha256':file_hash(bundle/'gaussians.ply'),'ui_verified':['food_click','highlight','same_id_after_view_change','orbit','background_no_food_name'],'screenshot':'data/outputs/fruit1_quality_demo_verified.png','remaining_failed_food_clicks':[c for c in report['clicks'] if c['expected_id']>0 and not c['correct']],'limitations':['Rendered images only; real photos not tested','SAM2 reference is not human ground truth','Held-out voting views were inspected during development; not an unbiased final test','3 of 72 food clicks still fail at view 000000','Foreground unknown pixel fraction is 5.94%, slightly above the 5.61% baseline']}
write_json(bundle/'validation.json',summary)
model['semantics_verified']=True
model['validation_scope']='rendered_input_development_views_with_known_limitations; see validation.json'
write_json(bundle/'model.json',model)
fusion['status']='quality_model_holdout_and_browser_checks_completed_with_limitations'
write_json(bundle/'semantics/fusion.json',fusion)
write_json(ProjectPaths().resolve('data/outputs/active_demo.json'),{'bundle':bundle.as_posix(),'validation':(bundle/'validation.json').as_posix(),'baseline_bundle':'data/outputs/fruit1_final'})
print('Active demo:',bundle)
