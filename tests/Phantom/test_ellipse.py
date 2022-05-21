from Phantom import *
import pytest
import numpy as onp


@pytest.mark.parametrize("n_theta", [100])
@pytest.mark.parametrize("n_p_prime", [200])
def test_compute_distance(n_theta, n_p_prime):
    theta = onp.linspace(0, 1, n_theta, endpoint=False) * onp.pi
    p_prime = onp.linspace(-0.5, 0.5, n_p_prime + 1, endpoint=True)

    ell: Ellipse = single_ellipse

    distance = compute_distance(ell.x_center, ell.y_center, ell.major_axis, ell.minor_axis, ell.angle,
                                p_prime, theta)

    assert distance.shape == (n_p_prime+1, n_theta)

