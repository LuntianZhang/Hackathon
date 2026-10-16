"""Record the reviewed video run and make it the local demo default."""
from pathlib import Path
from food3d.paths import ProjectPaths

from food3d.io import read_json, write_json
from food3d.reconstruction.gaussians import file_hash
from food3d.volume import load_calibration

bundle = ProjectPaths().resolve('data/outputs/fruit2_bundle')
model = read_json(bundle/'model.json')
fusion = read_json(bundle/'semantics/fusion.json')
evaluation = read_json(ProjectPaths().resolve('data/outputs/fruit2_evaluation/report.json'))
native = read_json(ProjectPaths().resolve('data/outputs/fruit2_native_parity/report.json'))
volumes = read_json(bundle/'measurements/volumes.json')
assert model['model_sha256'] == file_hash(bundle/'gaussians.npz')
assert all(item['model_sha256'] == model['model_sha256'] for item in [fusion, evaluation, volumes])
assert volumes['semantics_sha256'] == file_hash(bundle/'semantics/semantics.npz')
assert native['passed'] and evaluation['food_click_accuracy'] >= .95
assert evaluation['background_no_food_name_accuracy'] == 1
assert ProjectPaths().resolve('data/outputs/fruit2_demo_verified.png').is_file()
assert load_calibration(bundle, volumes) is None
inputs = read_json(ProjectPaths().resolve('data/outputs/fruit2_input_summary.json'))
inputs['naming_limitation'] = 'Small-fruit labels need confirmation; green Instance 2 predicted as mango'
write_json(ProjectPaths().resolve('data/outputs/fruit2_input_summary.json'), inputs)
validation = {
    'status': 'video_poc_checked_with_known_limitations',
    'input_summary': inputs, 'training_steps': 15000, 'gaussians': model['gaussians'],
    'test_count': 26, 'model_sha256': model['model_sha256'],
    'native_parity': native,
    'evaluation_report': 'data/outputs/fruit2_evaluation/report.json',
    'food_click_accuracy': evaluation['food_click_accuracy'],
    'mean_instance_iou': evaluation['mean_instance_iou'],
    'failed_food_clicks': [click for click in evaluation['clicks'] if click['expected_id'] > 0 and not click['correct']],
    'volume_unit': 'model_unit_cubed_no_metric_calibration',
    'ui_verified': ['banana_click_ID_3', 'food_highlight', 'orbit_near_reference_camera',
                    'same_ID_and_volume_across_views', 'green_fruit_ID_2', 'background_no_food_or_volume'],
    'screenshot': 'data/outputs/fruit2_demo_verified.png',
    'limitations': ['Food names for the small fruits require human confirmation',
                    'One banana click in frame 000040 was classified as background',
                    'Unobserved orbit angles have substantial reconstruction artifacts',
                    'SAM2 reference is not human ground truth; voting holdouts also used by SfM/SAM',
                    'No measured physical scale or true-volume validation']}
write_json(bundle/'validation.json', validation)
model['semantics_verified'] = True
model['validation_scope'] = 'video_development_consistency_and_UI_with_known_limitations; see validation.json'
write_json(bundle/'model.json', model)
fusion['status'] = 'video_holdout_and_browser_checks_completed_with_limitations'
write_json(bundle/'semantics/fusion.json', fusion)
write_json(ProjectPaths().resolve('data/outputs/active_demo.json'), {'bundle': bundle.as_posix(),
           'validation': (bundle/'validation.json').as_posix(), 'previous_bundle': 'data/outputs/fruit1_quality_cleaned'})
print('Active demo:', bundle)
