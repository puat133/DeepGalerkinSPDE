from Phantom import sinogram_matrix
import pytest
import numpy as onp


@pytest.mark.parametrize("n_theta", [5])
@pytest.mark.parametrize("n_p_prime", [5])
@pytest.mark.parametrize("fourier_basis_one_d", [4])
def test_sinogram_matrix(n_theta, n_p_prime, fourier_basis_one_d):
    thetas = onp.linspace(0, 1, n_theta, endpoint=False) * onp.pi
    p_prime = onp.linspace(-0.5, 0.5, n_p_prime + 1, endpoint=True)

    temp = onp.arange(-fourier_basis_one_d + 1, fourier_basis_one_d,
                      dtype=onp.int32)
    r, theta = onp.meshgrid(p_prime, thetas)
    ix, iy = onp.meshgrid(temp, temp)
    mat = sinogram_matrix(r.ravel(), theta.ravel(), ix.ravel(), iy.ravel())

    assert mat.shape == ((2*fourier_basis_one_d-1)**2, (n_p_prime+1)*n_theta)
