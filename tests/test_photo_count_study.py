import numpy as np
from PIL import Image
import pytest

from food3d.photos_only_volume import import_photos
from food3d.photo_count_study import calibrate_subset,canonical_groups,grouped_mask,merge_group_anchors
from food3d.schemas import UNKNOWN_ID
from food3d.synthetic import look_at


def test_requested_budgets_are_sampled_before_any_scene_metadata(tmp_path):
    source=tmp_path/'raw';source.mkdir()
    for index in range(91):Image.new('RGB',(8,8),(index,0,0)).save(source/f'{index:06d}.jpg')
    (source/'scene.json').write_bytes(b'must not be read')
    for count in (5,10,30,60,90):
        output=tmp_path/f'n{count}';photos=import_photos(source,output,count=count)
        assert len(photos)==count
        assert photos[0].path.read_bytes()==(source/'000000.jpg').read_bytes()
        assert photos[-1].path.read_bytes()==(source/'000090.jpg').read_bytes()
    with pytest.raises(ValueError):import_photos(source,tmp_path/'too_many',count=92)


def test_five_photos_fit_their_own_intrinsics_and_declare_low_evidence():
    from pansense.scan_geometry import marker_points
    k=np.array([[760.,0,320],[0,760.,240],[0,0,1]])
    points=np.concatenate([marker_points(i) for i in (20,22,25,27)]).astype(float)
    observations=[]
    for index,angle in enumerate(np.linspace(0,2*np.pi,5,endpoint=False)):
        center=np.array([.3*np.cos(angle),.3*np.sin(angle),.38+.22*np.cos(2*angle)])
        inverse=np.linalg.inv(look_at(center,np.zeros(3)))
        pixels=(points@inverse[:3,:3].T+inverse[:3,3])@k.T
        pixels=pixels[:,:2]/pixels[:,2:]
        observations.append({'frame_id':str(index),'width':640,'height':480,'marker_ids':[20,22,25,27],
            'objects':points.tolist(),'pixels':pixels.tolist()})
    calibration=calibrate_subset(observations)
    assert calibration['K'][0][0]==pytest.approx(760,rel=.001)
    assert calibration['small_sample_fit'] and calibration['views']==4
    assert calibration['heldout_frame_ids']==['0']
    assert not calibration['old_intrinsics_or_poses_used']
    with pytest.raises(ValueError,match='独立定位板标定视角'):
        calibrate_subset([observations[0]]*5)


def test_package_union_does_not_depend_on_instance_ids_or_partition():
    evidence={'objects':[{'instance_id':31,'name':'鼠标','label':'mouse','kind':'non_food'},
        {'instance_id':7,'name':'上包','label':'package','kind':'packaged_food'},
        {'instance_id':9,'name':'下包','label':'package','kind':'packaged_food'}]}
    labels=np.array([[0,31,7,9,UNKNOWN_ID]],np.uint16)
    result=grouped_mask(labels,canonical_groups(evidence))
    assert result.tolist()==[[0,1,2,2,UNKNOWN_ID]]
    merged={'objects':[evidence['objects'][0],dict(evidence['objects'][1],instance_id=11)]}
    labels[labels==7]=11;labels[labels==9]=11
    assert np.array_equal(result,grouped_mask(labels,canonical_groups(merged)))


def test_overlapping_package_seeds_are_unioned_before_becoming_unknown():
    mouse=np.array([[True,False,False,False]])
    top=np.array([[False,True,True,False]])
    lower=np.array([[False,False,True,True]])
    merged=merge_group_anchors({'f':{5:mouse,8:top,12:lower}},{1:[5],2:[8,12]})
    assert np.array_equal(merged['f'][1],mouse)
    assert merged['f'][2].tolist()==[[False,True,True,True]]
