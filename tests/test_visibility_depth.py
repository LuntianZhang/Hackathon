import pytest


def test_surface_depth_rejects_occluded_layer_bias_and_uncovered_rays():
    torch=pytest.importorskip('torch')
    from food3d.fusion.visibility import pixel_surface_depths
    pixels=torch.tensor([0,0,1,1,2])
    depths=torch.tensor([1.,10.,2.,3.,7.])
    mass=torch.tensor([.95,.045,.1,.8,0.])
    surface=pixel_surface_depths(pixels,depths,mass,4)
    torch.testing.assert_close(surface,torch.tensor([1.,3.,0.,0.]))
    expected_mean=(.95+10*.045)/(.95+.045)
    assert expected_mean>1.3  # Mean depth would wrongly reject the front surface.
