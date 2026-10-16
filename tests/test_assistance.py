from pathlib import Path

import numpy as np
import pytest

from food3d.segmentation.hints import validate_assistance, candidate_names, allowed_food
from food3d.segmentation.grounding import postprocess, suppress_boxes


def test_bilingual_aliases_deduplicate_food_classes_but_keep_rejection_alternatives():
    options = validate_assistance({'mode':'list', 'food_text':'苹果，apple\n香蕉;紫色圆果|purple fruit'})
    assert options['foods'] == [{'name':'苹果','text':'apple'}, {'name':'香蕉','text':'banana'},
                                {'name':'紫色圆果','text':'purple fruit'}]
    names = candidate_names(options, ['apple','bread'])
    assert len(names) == len(set(names))
    assert {'apple','banana','bread','table','plate','background','unknown'} <= set(names)
    assert not allowed_food(options, 'bread') and not allowed_food(options, 'background')
    assert allowed_food(options, 'banana')


@pytest.mark.parametrize('value', [
    {'mode':'list','food_text':''}, {'mode':'grounded','food_text':'无法翻译的自定义菜名'},
    {'mode':'list','food_text':'背景'}, {'mode':'oops'},
    {'mode':'manual','instances':[{'name':'苹果','point':[float('nan'), .5]}]},
    {'mode':'manual','instances':[{'name':'苹果','box':[.7,.2,.1,.8]}]},
    {'mode':'manual','instances':[{'name':'苹果','point':[.3,.5], 'box':[.1,.1,.5,.5]}]},
    {'mode':'manual','instances':[{'name':'苹果','point':[1.1,.5]}]},
    {'mode':'manual','instances':[{'name':'苹果','instance_id':1,'point':[.3,.5]},
                               {'name':'苹果','instance_id':1,'point':[.6,.5]}]},
])
def test_invalid_hints_cannot_start_expensive_model_work(value):
    with pytest.raises(ValueError): validate_assistance(value)


def test_two_apples_are_independent_instances_with_user_confirmed_names():
    options = validate_assistance({'mode':'manual','instances':[
        {'instance_id':7,'name':'苹果','point':[.3,.5]},
        {'instance_id':12,'name':'苹果','box':[.6,.2,.9,.8]}]})
    assert [r['instance_id'] for r in options['instances']] == [7,12]
    assert [r['name'] for r in options['instances']] == ['苹果','苹果']


def test_grounding_api_uses_the_installed_threshold_argument_without_breaking_older_api():
    class New:
        def post_process_grounded_object_detection(self, outputs, input_ids, threshold, text_threshold, target_sizes):
            return [{'threshold':threshold,'size':target_sizes}]
    class Old:
        def post_process_grounded_object_detection(self, outputs, input_ids, box_threshold, text_threshold, target_sizes):
            return [{'threshold':box_threshold,'size':target_sizes}]
    for processor in (New(), Old()):
        assert postprocess(processor,None,None,(640,480)) == {'threshold':.3,'size':[(480,640)]}


def test_grounding_overlapping_category_boxes_do_not_create_duplicate_instances():
    result = suppress_boxes([{'box':[10,10,40,40],'grounding_score':.8,'name':'苹果'},
        {'box':[11,11,40,40],'grounding_score':.7,'name':'梨'},
        {'box':[70,10,95,40],'grounding_score':.65,'name':'苹果'}])
    assert len(result) == 2 and {r['name'] for r in result} == {'苹果'}
