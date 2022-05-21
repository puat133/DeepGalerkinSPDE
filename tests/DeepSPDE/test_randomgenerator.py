import jax.numpy as np
from DeepSPDE import RandomGenerator
import pytest
import os
# os.environ["CUDA_VISIBLE_DEVICES"]=""


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("essential_basis_number", [16, 32])
@pytest.mark.parametrize("prngkey_num", [0, 3])
@pytest.mark.parametrize("n", [None, 1000])
def test_random_generator(essential_basis_number: int, dimension: int, prngkey_num: int, n: int):
    rg = RandomGenerator(essential_basis_number, dimension, prngkey_num)

    if dimension == 1:
        true_dimension = essential_basis_number
    else:
        true_dimension = 2 * essential_basis_number * essential_basis_number - 2 * essential_basis_number + 1

    if not n:
        w_half_test = rg.construct_w_half()
        assert w_half_test.shape[0] == true_dimension
        assert w_half_test[0].imag == 0

        w_half_test_second = rg.construct_w_half()
        assert not np.array_equal(w_half_test, w_half_test_second)

        w_test = rg.construct_w()
        assert w_test.shape[0] == 2 * true_dimension - 1
        assert w_test[true_dimension - 1].imag == 0

        w_test_second = rg.construct_w()
        assert not np.array_equal(w_test, w_test_second)
    else:
        w_half_test = rg.construct_w_half(n)
        assert w_half_test.shape[1] == true_dimension
        assert np.all(w_half_test[:, 0].imag == 0)

        assert np.abs(np.cov(w_half_test[:, 0]).real - 1) < 1e-1

        w_half_test_second = rg.construct_w_half(n)
        assert not np.array_equal(w_half_test, w_half_test_second)

        w_test = rg.construct_w(n)
        assert w_test.shape[1] == 2 * true_dimension - 1
        assert np.all(w_test[:, true_dimension - 1].imag == 0)

        w_test_second = rg.construct_w(n)
        assert not np.array_equal(w_test, w_test_second)
        assert np.abs(np.cov(w_test[:, true_dimension - 1]).real - 1) < 1e-1
