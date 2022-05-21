import jax.numpy as np
from utils.hdf_io import save_to_hdf
from DeepSPDE import Fourier, LMatrixGenerator, RandomGenerator
from Measurements import OneDSampleSignalShape
from Optimizers import OptimizerType


def test_save_to_hdf():
    x = np.arange(1000)
    m, n = np.meshgrid(x, x)
    fo = Fourier(16, 32, 2)
    el_mat_gen = LMatrixGenerator(f=fo)
    rg = RandomGenerator(fo.essential_basis_number, fo.dimension, 0)
    a = "laksdjlaksjdlkjsa"
    b = OneDSampleSignalShape.SMOOTH_DISCONTINUOUS
    c = OptimizerType.RMSPROP

    save_to_hdf('./test.hdf5', locals())
