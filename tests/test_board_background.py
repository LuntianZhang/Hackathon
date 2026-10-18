from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
import pycolmap
import pytest

from food3d.data.board_background import foreground_mask,undistort_mask,prepare_background,feature_mask_path,apply_background
from food3d.io import read_json,write_json
from food3d.schemas import CameraFrame


def test_preserves_tall_object_outside_board_and_excludes_background():
    polygon=[[40,40],[80,60],[60,85],[20,65]]
    labels=np.zeros((100,100),np.uint16);labels[10:65,45:55]=7;labels[20,55]=65535
    mask=foreground_mask((100,100),polygon,labels)
    assert mask[20,50]==255 and mask[20,55]==255 and mask[65,50]==255
    assert mask[5,5]==0 and mask[90,90]==0


@pytest.mark.parametrize('model,params',[
    ('PINHOLE',[70,75,40,30]),('SIMPLE_RADIAL',[70,40,30,.12])])
def test_undistortion_uses_actual_camera_and_half_pixel_convention(model,params):
    camera=pycolmap.Camera(model=model,width=80,height=60,params=params)
    mask=np.zeros((60,80),np.uint8);mask[5:45,10:55]=255
    k=np.array([[70,0,39.5],[0,75 if model=='PINHOLE' else 70,29.5],[0,0,1]])
    result=undistort_mask(mask,camera,k,(80,60))
    distortion=np.array([.12,0,0,0]) if model=='SIMPLE_RADIAL' else np.zeros(4)
    x,y=cv2.initUndistortRectifyMap(k,distortion,None,k,(80,60),cv2.CV_32FC1)
    expected=cv2.remap(mask,x,y,cv2.INTER_NEAREST)
    assert np.array_equal(result,expected)


def raw_scene(tmp_path):
    scene=tmp_path/'scene';(scene/'frames').mkdir(parents=True)
    frames=[]
    for i in range(3):
        Image.new('RGB',(100,100),(50+i*10,80,100)).save(scene/'frames'/f'{i:06d}.jpg')
        frames.append({'frame_id':f'{i:06d}','image_path':f'frames/{i:06d}.jpg','accepted':True})
    write_json(scene/'frame_manifest.json',{'frames':frames})
    return scene


def test_background_preparation_needs_no_poses_and_never_falls_back_to_full_frame(tmp_path,monkeypatch):
    import food3d.data.board_background as background
    import food3d.segmentation.llm_segment as llm
    import food3d.segmentation.vision_llm as vision
    scene=raw_scene(tmp_path)
    polygon=[[20,40],[80,40],[80,80],[20,80]]
    monkeypatch.setattr(background,'board_polygon',lambda image:None if image.getpixel((0,0))[0]>65 else
        {'polygon_pixels':polygon,'marker_ids':[20,21,22]})
    monkeypatch.setattr(vision.VisionSettings,'load',lambda:SimpleNamespace(max_views=2))
    monkeypatch.setattr(vision,'identify_views',lambda *a,**k:{'objects':[{'instance_id':7}]})
    monkeypatch.setattr(llm,'anchor_masks',lambda *a,**k:{})
    def track(frames,anchors,output,checkpoint,identifiers):
        assert all(not hasattr(f,'camera_to_world') for f in frames)
        for frame in frames:
            labels=np.zeros((100,100),np.uint16);labels[5:55,45:55]=7
            Image.fromarray(labels).save(output/'masks'/f'{frame.frame_id}.png')
    monkeypatch.setattr(llm,'track_anchors',track)
    result=prepare_background(scene,Path('unused.pt'),minimum=2)
    assert result['processed_frames']==2 and result['excluded_frames']==1
    assert not result['uses_camera_poses'] and not (scene/'scene.json').exists()
    manifest=read_json(scene/'frame_manifest.json')
    assert not manifest['frames'][2]['accepted'] and not (scene/'frames/000002.jpg').exists()
    root=feature_mask_path(scene,manifest)
    mask=np.asarray(Image.open(root/'000000.jpg.png'))
    assert mask[10,50] and not mask[5,5]
    assert (scene/'board_background/debug/preview.jpg').is_file()
    (root/'000000.jpg.png').unlink()
    with pytest.raises(ValueError,match='缺失'):feature_mask_path(scene,manifest)


def test_background_uses_one_pass_only_with_repeated_end_anchors(tmp_path,monkeypatch):
    import food3d.data.board_background as background
    import food3d.segmentation.llm_segment as llm
    import food3d.segmentation.vision_llm as vision
    scene=raw_scene(tmp_path)
    monkeypatch.setattr(background,'board_polygon',lambda image:
        {'polygon_pixels':[[10,10],[90,10],[90,90],[10,90]],'marker_ids':[20,21,22]})
    monkeypatch.setattr(vision.VisionSettings,'load',lambda:SimpleNamespace(max_views=3))
    monkeypatch.setattr(vision,'identify_views',lambda *a,**k:{'model':'test','keyframes':['000000','000001','000002'],
        'objects':[{'instance_id':7,'name':'苹果','label':'apple','kind':'food','observations':[]}]})
    monkeypatch.setattr(llm,'anchor_masks',lambda *a,**k:{fid:{7:np.ones((100,100),bool)}
        for fid in ('000000','000001','000002')})
    calls=[]
    def track(frames,anchors,output,checkpoint,identifiers,**kwargs):
        calls.append(kwargs)
        for frame in frames:
            Image.fromarray(np.full((100,100),7,np.uint16)).save(output/'masks'/f'{frame.frame_id}.png')
    monkeypatch.setattr(llm,'track_anchors',track)
    result=prepare_background(scene,Path('unused.pt'),minimum=2)
    assert result['tracking_passes']=='forward_only'
    assert calls==[{'reverse_pass':False}]


def test_missing_boards_fail_before_any_model_call_or_manifest_mutation(tmp_path,monkeypatch):
    import food3d.data.board_background as background
    scene=raw_scene(tmp_path);before=(scene/'frame_manifest.json').read_bytes()
    monkeypatch.setattr(background,'board_polygon',lambda image:None)
    with pytest.raises(ValueError,match='关闭排除背景'):
        prepare_background(scene,Path('unused.pt'),minimum=2)
    assert before==(scene/'frame_manifest.json').read_bytes()
    assert len(list((scene/'frames').iterdir()))==3


def test_training_images_preserve_foreground_and_geometry_and_keep_originals(tmp_path):
    scene=tmp_path/'scene';(scene/'frames').mkdir(parents=True);(scene/'images').mkdir()
    picture=np.full((60,80,3),[20,70,110],np.uint8)
    Image.fromarray(picture).save(scene/'images/000000.png')
    Image.fromarray(picture).save(scene/'frames/000000.png')
    root=scene/'board_background/feature_masks';root.mkdir(parents=True)
    mask=np.zeros((60,80),np.uint8);mask[10:40,20:50]=255
    Image.fromarray(mask).save(root/'000000.png.png')
    manifest={'frames':[{'image_path':'frames/000000.png','accepted':True}],
        'board_background':{'enabled':True,'feature_mask_path':'board_background/feature_masks'}}
    native=tmp_path/'native';native.mkdir()
    (native/'cameras.txt').write_text('1 PINHOLE 80 60 70 75 40 30\n')
    (native/'images.txt').write_text('1 1 0 0 0 0 0 0 1 000000.png\n\n')
    (native/'points3D.txt').write_text('')
    frame=CameraFrame('000000','images/000000.png',80,60,
        [[70,0,39.5],[0,75,29.5],[0,0,1]],np.eye(4).tolist())
    updated=apply_background(scene,manifest,[frame],native)
    result=np.asarray(Image.open(scene/updated[0].image_path))
    assert np.array_equal(result[20,30],picture[20,30]) and (result[0,0]==255).all()
    assert updated[0].K==frame.K and updated[0].camera_to_world==frame.camera_to_world
    assert np.array_equal(np.asarray(Image.open(scene/frame.image_path)),picture)


def test_completed_background_tracking_reuses_instances_after_undistortion(tmp_path,monkeypatch):
    from food3d.segmentation.llm_segment import segment_llm
    from food3d.schemas import save_scene
    scene=tmp_path/'scene';raw=scene/'frames';raw.mkdir(parents=True)
    image=np.full((60,80,3),180,np.uint8)
    Image.fromarray(image).save(raw/'000000.jpg')
    original=pycolmap.Camera(model='SIMPLE_RADIAL',width=80,height=60,params=[70,40,30,.12])
    frame=CameraFrame('000000','frames/000000.jpg',80,60,
        [[70,0,39.5],[0,70,29.5],[0,0,1]],np.eye(4).tolist())
    save_scene(scene/'scene.json',[frame],source='colmap',world_units='arbitrary_sfm')
    write_json(scene/'frame_manifest.json',{'frames':[{'frame_id':'000000','image_path':'frames/000000.jpg','accepted':True}],
        'board_background':{'status':'applied_to_sfm_and_training'}})
    model=scene/'colmap/sparse/1';model.mkdir(parents=True)
    write_json(scene/'colmap_selection.json',{'selected_model':'colmap/sparse/1'})
    monkeypatch.setattr(pycolmap,'Reconstruction',lambda path:SimpleNamespace(
        images={1:SimpleNamespace(name='000000.jpg',camera_id=1)},cameras={1:original}))
    background=scene/'board_background'
    for folder in ('masks','confidence'): (background/folder).mkdir(parents=True)
    labels=np.zeros((60,80),np.uint16);labels[12:40,20:52]=7
    Image.fromarray(labels).save(background/'masks/000000.png')
    np.save(background/'confidence/000000.npy',np.where(labels==7,.9,1).astype(np.float16))
    write_json(background/'vision_evidence.json',{'model':'test','keyframes':['000000'],
        'objects':[{'instance_id':7,'name':'苹果','label':'apple','kind':'food',
                    'observations':[{'frame_id':'000000','sam_score':.95}]}]})
    output=tmp_path/'segmentation'
    result=segment_llm(scene,output,tmp_path/'missing_sam_weights.pt','000000')
    assert result['initialization']=='board_background_reuse'
    assert result['pixel_counts']['000000']['7']>0
    assert read_json(output/'instances.json')['instances'][1]['name']=='苹果'
    actual=np.asarray(Image.open(output/'masks/000000.png'))
    assert np.array_equal(actual,undistort_mask(labels,original,frame.K,(80,60)))
    assert np.load(output/'confidence/000000.npy').shape==(60,80)
    assert (output/'debug/mask_sheet.png').is_file()


def test_actual_colmap_feature_extraction_honors_background_masks(tmp_path):
    images=tmp_path/'frames';images.mkdir();masks=tmp_path/'masks';masks.mkdir()
    rng=np.random.default_rng(7)
    pixels=cv2.GaussianBlur(rng.integers(0,256,(256,256),dtype=np.uint8),(3,3),.7)
    Image.fromarray(pixels).save(images/'000000.png')
    mask=np.zeros((256,256),np.uint8);mask[:,128:]=255
    Image.fromarray(mask).save(masks/'000000.png.png')
    reader=pycolmap.ImageReaderOptions();reader.mask_path=masks
    extraction=pycolmap.FeatureExtractionOptions();extraction.use_gpu=False;extraction.num_threads=1
    database=tmp_path/'database.db'
    pycolmap.extract_features(database,images,reader_options=reader,extraction_options=extraction,device=pycolmap.Device.cpu)
    with pycolmap.Database.open(database) as db:
        image=db.read_all_images()[0];keypoints=db.read_keypoints(image.image_id)
    assert len(keypoints)>0 and np.all(keypoints[:,0]>=127.5)
