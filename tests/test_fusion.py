import numpy as np

from food3d.fusion.visibility import front_to_back_weights
from food3d.fusion.voting import finalize_votes, weight_view_evidence
from food3d.schemas import UNKNOWN_ID
from food3d.viewer.app import describe_click


def test_occluded_gaussian_receives_only_transmitted_contribution():
    pixels = np.array([1, 0, 1, 0])
    depths = np.array([3, 2, 2, 1])
    alphas = np.array([0.9, 0.9, 0.9, 0.9])
    weights = front_to_back_weights(pixels, depths, alphas)
    np.testing.assert_allclose(weights, [0.09, 0.09, 0.9, 0.9])
    opaque_front = alphas.copy()
    opaque_front[3] = 0.999
    adjusted = front_to_back_weights(pixels, depths, opaque_front)
    assert adjusted[1] < 0.001
    assert adjusted[2] == weights[2]


def test_fusion_keeps_background_and_rejects_conflicts_or_single_view():
    votes = np.array([[3, 0, 0], [0.2, 4, 0.2], [0, 1, 1], [0, 2, 0], [0, 0, 0]], dtype=float)
    support = np.array([[3, 0, 0], [0, 5, 0], [0, 3, 3], [0, 1, 0], [0, 0, 0]])
    labels, confidence, views = finalize_votes(votes, support, np.array([0, 1, 2]))
    np.testing.assert_array_equal(labels, [0, 1, UNKNOWN_ID, UNKNOWN_ID, UNKNOWN_ID])
    assert confidence[1] > 0.9
    assert confidence[-1] == 0


def test_click_uses_semantic_ids_and_never_names_background():
    labels = np.array([[0, 2, UNKNOWN_ID]], dtype=np.uint16)
    confidence = np.array([[1, 0.9, 0.9]])
    names = {2: "苹果"}
    instance, description = describe_click(labels, confidence, 1, 0, names)
    assert instance == 2 and "苹果" in description
    for x in [0, 2]:
        instance, description = describe_click(labels, confidence, x, 0, names)
        assert instance == 0 and "苹果" not in description


def test_fine_low_opacity_splats_get_support_but_occluded_residue_does_not():
    # Three repeated views: small visible food splat, negligible rear residue,
    # and an edge straddling equal background/food evidence.
    raw = np.array([[0,0.002],[0,0.00001],[0.01,0.01]],dtype=np.float32)
    weights,support = weight_view_evidence(raw)
    labels,confidence,views = finalize_votes(3*weights,3*support,np.array([0,1]))
    np.testing.assert_array_equal(labels,[1,UNKNOWN_ID,UNKNOWN_ID])
    assert confidence[0]==1 and views[0]==3
    assert weights[0,1]<1  # Weak contributions retain their lower vote weight.
