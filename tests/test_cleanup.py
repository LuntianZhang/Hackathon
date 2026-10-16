import numpy as np
from food3d.reconstruction.cleanup import background_floater_mask
from food3d.schemas import UNKNOWN_ID


def test_cleanup_removes_isolated_opaque_background_but_preserves_food_and_boundaries():
    labels=np.array([UNKNOWN_ID]*5+[0])
    opacity=np.array([.99,.99,.99,.4,.99,.99])
    confidence=np.array([1.,1.,.5,1.,1.,1.])
    support=np.array([2,2,2,2,1,8])
    votes=np.array([[2,0],[0,2],[1,1],[2,0],[.5,0],[8,0]],dtype=float)
    removed=background_floater_mask(opacity,labels,confidence,support,votes,np.array([0,1]))
    np.testing.assert_array_equal(removed,[True,False,False,False,False,False])
