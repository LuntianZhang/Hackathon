import numpy as np

from food3d.data.view_sampling import orbit_coordinates, select_uniform_views, select_common_holdout, fit_similarity


def circle_poses(count):
    poses=[]
    for angle in np.linspace(0, 2*np.pi, count, endpoint=False):
        center=np.array([np.cos(angle),np.sin(angle),.4])
        forward=-center/np.linalg.norm(center)
        right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right)
        down=np.cross(forward,right)
        pose=np.eye(4);pose[:3,:3]=np.column_stack([right,down,forward]);pose[:3,3]=center
        poses.append(pose)
    return poses


def test_uniform_azimuths_unique_and_disjoint_holdout():
    orbit=orbit_coordinates(circle_poses(120))
    assert orbit['closed']
    selected, distribution=select_uniform_views(orbit,5)
    assert len(set(selected))==5
    assert distribution['max_angular_error_degrees']<1e-6
    phase=np.sort(orbit['angles'][selected])
    gaps=np.diff(np.r_[phase,phase[0]+2*np.pi])
    np.testing.assert_allclose(np.degrees(gaps),72,atol=1e-6)
    remaining=[i for i in range(120) if i not in selected]
    holdout,_=select_uniform_views(orbit,10,remaining)
    assert not set(holdout)&set(selected)
    holdout,info=select_common_holdout(orbit,selected,count=10,minimum_degrees=3)
    assert not set(holdout)&set(selected)
    assert info['minimum_distance_to_any_training_view_degrees']>=3-1e-8


def test_camera_alignment_recovers_rotation_scale_and_translation():
    source=np.array(circle_poses(12))[:,:3,3]
    expected=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    target=source@expected.T*2.5+np.array([3,-4,2])
    rotation,scale,translation,error=fit_similarity(source,target)
    np.testing.assert_allclose(rotation,expected,atol=1e-8)
    np.testing.assert_allclose(translation,[3,-4,2],atol=1e-8)
    assert abs(scale-2.5)<1e-8 and error<1e-8
