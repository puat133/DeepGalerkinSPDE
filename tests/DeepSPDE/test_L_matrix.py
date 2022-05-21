import jax.numpy as np
import pytest
from DeepSPDE import Fourier, RandomGenerator, LMatrixGenerator
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


@pytest.mark.parametrize("essential_basis_number", [32, 64])
@pytest.mark.parametrize("multiplier", [1, 2])
@pytest.mark.parametrize("dimension", [1])
def test_toeplitz_form(essential_basis_number: int, multiplier: int, dimension: int):
    fo = Fourier(essential_basis_number, multiplier * essential_basis_number, dimension)
    x = np.arange(2 * fo.target_basis_number - 1)
    y = np.sin(2 * np.pi * x / fo.target_basis_number)
    u_h = fo.rfft(y)
    el_mat_gen = LMatrixGenerator(f=fo)
    res = el_mat_gen.toeplitz_form(u_h)

    assert res.ndim == 2
    assert res.shape[0] == res.shape[1]
    count_nonzero = (2 * fo.basis_number - 1) ** 2 - fo.basis_number * (fo.basis_number - 1)
    assert np.allclose(np.count_nonzero(res), count_nonzero)


@pytest.mark.parametrize("essential_basis_number", [16])
@pytest.mark.parametrize("multiplier", [1, 2])
@pytest.mark.parametrize("dimension", [2])
@pytest.mark.parametrize("n_prior_layers", [2])
def test_(dimension: int, essential_basis_number: int, multiplier: int, n_prior_layers: int):
    fo = Fourier(essential_basis_number, multiplier * essential_basis_number, dimension)
    rg = RandomGenerator(fo.essential_basis_number, fo.dimension)
    l_matrix = LMatrixGenerator(f=fo)
    l_matrix.kappa0 = 1e0

    w_halfs = rg.construct_w_half(n_prior_layers)
    sigmas = np.ones(n_prior_layers + 1, dtype=np.float32)
    l_res = l_matrix.generate_last_l_matrix_from_u_half_sequence(w_halfs, sigmas)

    if dimension == 1:
        count_nonzero = (2 * fo.essential_basis_number - 1) ** 2 \
                        - fo.essential_basis_number * (fo.essential_basis_number - 1)
        assert np.allclose(np.count_nonzero(l_res), count_nonzero)

    if dimension == 2:
        count_nonzero_one_d = (2 * fo.essential_basis_number - 1) ** 2 \
                              - fo.essential_basis_number * (fo.essential_basis_number - 1)

        count_nonzero = count_nonzero_one_d * count_nonzero_one_d
        assert np.allclose(np.count_nonzero(l_res), count_nonzero)
        pass
