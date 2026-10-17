from io import BytesIO

import cv2
import numpy as np
from PIL import Image
import pytest

from food3d.segmentation.board_scope import board_polygon,filter_to_board,scoped_result


def board_photo():
    from pansense.scan_geometry import square_board_png
    board=Image.open(BytesIO(square_board_png(80))).convert('RGB')
    side=board.width
    source=np.float32([[0,0],[side-1,0],[side-1,side-1],[0,side-1]])
    target=np.float32([[380,50],[750,350],[440,750],[70,390]])
    matrix=cv2.getPerspectiveTransform(source,target)
    return Image.fromarray(cv2.warpPerspective(np.asarray(board),matrix,(800,800),borderValue=(210,210,210)))


def obj(identifier,x,y,frames=('a',)):
    return {'instance_id':identifier,'name':f'对象 {identifier}','label':'test object','kind':'non_food',
        'observations':[{'frame_id':fid,'box':[x-.02,y-.02,x+.02,y+.02],'confidence':.9} for fid in frames]}


def test_perspective_board_filters_outside_polygon_not_just_bounding_rectangle():
    photo=board_photo();region=board_polygon(photo)
    assert region is not None and region['layout']=='square_20cm'
    inside=obj(7,.5,.5);outside=obj(9,.15,.15)
    objects,scope=filter_to_board([inside,outside],[('a',photo)])
    assert [o['instance_id'] for o in objects]==[7]
    assert scope['status']=='checked' and scope['objects'][1]['outside_views']==1
    assert len(outside['observations'])==1  # original model evidence is preserved


def test_unknown_board_views_do_not_rescue_a_confirmed_outside_object():
    photo=board_photo();blank=Image.new('RGB',(800,800),'gray')
    objects,scope=filter_to_board([obj(9,.15,.15,('a','b'))],[('a',photo),('b',blank)])
    assert objects==[] and scope['objects'][0]['unverified_views']==1
    with pytest.raises(ValueError,match='白色定位板内'):
        scoped_result({'objects':[obj(9,.15,.15)]},[('a',photo)])


def test_no_verified_markers_keeps_model_board_only_scope_explicit():
    objects,scope=filter_to_board([obj(7,.5,.5)],[('a',Image.new('RGB',(80,80),'white'))])
    assert [o['instance_id'] for o in objects]==[7]
    assert scope['status']=='prompt_only'
