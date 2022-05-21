import pytest
import jax.numpy as np
from utils import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("essential_basis_number", [2, 4])
@pytest.mark.parametrize("multi", [False, True])
def test_symmetrize(essential_basis_number: int, dimension: int, multi: bool):
    if dimension == 1:
        w_half_test = np.arange(essential_basis_number) + 1j * np.arange(essential_basis_number)
        w_true = np.concatenate((np.flip(w_half_test[1:]).conj(), w_half_test))
        w_res = symmetrize(w_half_test, False)

        assert w_res.ndim == 1
        assert w_res.shape[0] == 2 * w_half_test.shape[0] - 1
        assert np.array_equal(w_true, w_res)

    if dimension == 2:
        if not multi:
            temp = np.arange(essential_basis_number * (2 * essential_basis_number - 1))
            temp = temp.reshape(2 * essential_basis_number - 1, essential_basis_number)
            w_half_test = temp + 1j * temp

            w_res = symmetrize(w_half_test, False)

            assert w_res.ndim == 2
            assert w_res.shape[0] == 2 * w_half_test.shape[1] - 1
            assert w_res.shape[1] == 2 * w_half_test.shape[1] - 1
        else:
            w_half_ = np.arange(essential_basis_number) + 1j * np.arange(essential_basis_number)
            w_half_test = np.tile(w_half_, (2, 1))
            w_sym_ = np.concatenate((np.flip(w_half_[1:]).conj(), w_half_))
            w_sym_true = np.tile(w_sym_, (2, 1))
            w_res = symmetrize(w_half_test, True)

            assert w_res.ndim == 2
            assert w_res.shape[0] == 2
            assert w_res.shape[1] == 2 * w_half_test.shape[1] - 1
            assert np.array_equal(w_sym_true, w_res)


@pytest.mark.parametrize("dimension", [1, 2])
@pytest.mark.parametrize("essential_basis_number", [2, 16])
def test_padding(essential_basis_number: int, dimension: int):
    if dimension == 1:
        target_basis_number = 2 * essential_basis_number
        w_half_test = np.arange(essential_basis_number) + 1j * np.arange(essential_basis_number)
        w_res = pad(w_half_test, target_basis_number)

        padded_num = target_basis_number - essential_basis_number
        assert w_res.ndim == 1
        assert w_res.shape[0] == target_basis_number
        assert np.array_equal(w_half_test, w_res[:-padded_num])
        assert np.array_equal(np.zeros((target_basis_number - essential_basis_number)),
                               w_res[essential_basis_number:])

    if dimension == 2:
        target_basis_number = 2 * essential_basis_number
        temp = np.arange(essential_basis_number * (2 * essential_basis_number - 1))
        temp = temp.reshape(2 * essential_basis_number - 1, essential_basis_number)
        w_half_test = temp + 1j * temp

        w_res = pad(w_half_test, target_basis_number)

        padded_num = target_basis_number - essential_basis_number
        assert w_res.ndim == 2
        assert w_res.shape[0] == 2 * target_basis_number - 1
        assert w_res.shape[1] == target_basis_number
        assert np.array_equal(w_half_test, w_res[padded_num:-padded_num, :-padded_num])
        assert not np.any(w_res[:padded_num, :])
        assert not np.any(w_res[-padded_num:, :])
        assert not np.any(w_res[:, -padded_num:])
