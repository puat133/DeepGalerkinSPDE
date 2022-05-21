import numpy as onp
import Measurements.measurement_base as m_base
from Phantom import Ellipse, sinogram_of_ellipses, sinogram_matrix
from DeepSPDE import Fourier
from typing import List


class Sinogram(m_base.LinearMeasurementBase):
    """
    Sinogram measurement class which encapsulates the sinogram. Notice that here we specifically use
    a collection of many ellipses, and obtain the radon transform analytically (not via discrete Radon
    transform as in skimage).
    """

    def __init__(self, ellipses: List[Ellipse], fourier: Fourier, thetas: onp.ndarray, numpoints: int, std_dev: float):
        """
        The instantiation.

        Parameters
        ----------
        ellipses: List[Ellipse]
            Collection of ellipses.
        fourier: Fourier
            Fourier transform object.
        thetas: onp.ndarray
            Projection angle.
        numpoints: int
            Step point of the integration variable.
        std_dev: float
            Standard deviation of the measurement noise.
        """
        self.__thetas = thetas
        self.__ellipses = ellipses
        self.__numpoints = numpoints
        self.__p_range = onp.linspace(-0.5, 0.5, self.__numpoints, endpoint=True)
        self.__std_dev = std_dev
        self.__ground_truth = None
        self.__ground_truth_computed = False
        self.__fourier = fourier
        self.__measurement_matrix_computed = False
        self.__measurement_matrix = None
        self.__ravel_order = 'F'

    @property
    def std_dev(self):
        return self.__std_dev

    @property
    def numpoints(self):
        return self.__numpoints

    @property
    def thetas(self):
        return self.__thetas

    @property
    def ellipses(self):
        return self.__ellipses

    @property
    def numpoints(self):
        return self.__numpoints

    @property
    def range(self):
        return self.__p_range

    @property
    def ravel_order(self):
        return self.__ravel_order

    @property
    def measurement_matrix(self) -> onp.ndarray:
        if not self.__measurement_matrix_computed:
            temp = onp.arange(-self.__fourier.essential_basis_number + 1, self.__fourier.essential_basis_number,
                              dtype=onp.int32)
            theta, r = onp.meshgrid(self.__thetas, self.__p_range)
            ix, iy = onp.meshgrid(temp, temp)
            self.__measurement_matrix = sinogram_matrix(r.ravel(self.__ravel_order), theta.ravel(self.__ravel_order),
                                                        ix.ravel(self.__ravel_order), iy.ravel(self.__ravel_order))
            self.__measurement_matrix_computed = True
        else:
            return self.__measurement_matrix

    def measure(self) -> onp.ndarray:
        meas_error = self.__std_dev * onp.random.randn(*self.ground_truth.shape)
        return self.ground_truth + meas_error

    @property
    def ground_truth(self) -> onp.ndarray:
        if not self.__ground_truth_computed:
            self.__ground_truth = sinogram_of_ellipses(self.__ellipses, self.__p_range, self.__thetas)
            self.__ground_truth_computed = True
        return self.__ground_truth
