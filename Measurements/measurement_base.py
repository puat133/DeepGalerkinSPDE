from abc import ABC, abstractmethod
import numpy as onp


class LinearMeasurementBase(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def measurement_matrix(self) -> onp.ndarray:
        """
        Abstract method that gives the measurement matrix corresponds to
        a linear measurement.

        Returns
        -------
        out: numpy.ndarray
            A measurement matrix
        """
        pass

    @abstractmethod
    def measure(self) -> onp.ndarray:
        """
        Abstract method that gives one sample of measurement.
        Returns
        -------
        out: numpy.ndarray
            Measurement sample.
        """
        pass

    @property
    @abstractmethod
    def ground_truth(self) -> onp.ndarray:
        """
        The measurement without noise added
        Returns
        -------
        out: numpy.ndarray
            Ground truth.
        """
        pass
