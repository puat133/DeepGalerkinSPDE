import jax.numpy as np
import pytest
from DeepSPDE import Fourier
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""


@pytest.mark.parametrize("essential_basis_number", [32, 64])
@pytest.mark.parametrize("multiplier", [1, 2, 4])
@pytest.mark.parametrize("dimension", [1, 2])
def test_fourier(essential_basis_number: int, multiplier: int, dimension: int):
    if dimension == 1:
        fo = Fourier(essential_basis_number, multiplier * essential_basis_number, dimension)
        x = np.arange(2 * fo.target_basis_number - 1)
        y = np.sin(2 * np.pi * x / (fo.target_basis_number - 1))
        y_hat = fo.rfft(y)
        assert y_hat.shape[0] == fo.basis_number
        y_ = fo.irfft(y_hat)
        assert y_.shape == y.shape
        if multiplier == 1:
            assert np.allclose(y, y_, atol=1e-5)
        else:
            assert np.linalg.norm(y - y_) < 1.e-1

    if dimension == 2:
        fo = Fourier(essential_basis_number, multiplier * essential_basis_number, dimension)
        x = np.arange(2 * fo.target_basis_number - 1)
        x_grid, y_grid = np.meshgrid(x, x)
        z = np.sin(2 * np.pi * (x_grid + y_grid) / (fo.target_basis_number - 1))

        z_hat = fo.rfft(z)
        assert (2 * fo.essential_basis_number - 1, fo.essential_basis_number) == z_hat.shape

        z_ = fo.irfft(z_hat)
        assert z_.shape == z.shape
        if multiplier == 1:
            assert np.allclose(z, z_, atol=1e-2)
        else:
            assert np.linalg.norm(z - z_) < 1.
        pass
