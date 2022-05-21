from enum import Enum
import h5py
import pathlib
import numpy as onp

import Measurements.measurement_base as m_base


class OneDSampleSignalShape(Enum):
    SMOOTH_DISCONTINUOUS = 0
    BOX = 1
    RECT = 2
    RECT2 = 3


def _eigen_function_one_d(i: int, t: onp.ndarray) -> onp.ndarray:
    """
    one dimensional eigen function of a Laplacian operator in (0,1)

    Parameters
    ----------
    i : int
        Frequency.
    t : numpy.ndarray
        Time signal from (0,1).

    Returns
    -------
    out: numpy.ndarray
        The function output.
    """
    return onp.exp(2 * onp.pi * 1j * i * t)


class DenoisingOneD(m_base.LinearMeasurementBase):
    def __init__(self, num_points: int,
                 basis_number: int,
                 std_dev: float,
                 save_to_disk: bool = True,
                 save_path: str = './.measurements/',
                 number_of_sample: int = 1,
                 random_seed: int=0,
                 signal_shape: OneDSampleSignalShape = OneDSampleSignalShape.SMOOTH_DISCONTINUOUS):
        """
        Create DenoisingOneD instance.

        Parameters
        ----------
        num_points: int
            Number of measurement points
        basis_number: int
            Number of Fourier basis which will be used later on in constructing the measurement matrix.
        std_dev: float
            Measurement noise standard deviation.
        save_to_disk: bool
            Whether to save to disk or not.
        save_path: str
            Relative path string to save the measurement samples.
        number_of_sample: int
            Number of measurement samples generated.
        random_seed: int

        signal_shape: OneDSampleSignalShape
            Signal shape.
        """
        super().__init__()
        self.__num_points = num_points
        self.__basis_number = basis_number
        self.__signal_shape = signal_shape
        self.__std_dev = std_dev
        self.__t = onp.linspace(0, 1., self.__num_points, dtype=onp.float32)
        self.__ground_truth_signal = onp.zeros_like(self.__t, dtype=onp.float32)
        self.__measurement_matrix = self.__measurement_matrix = onp.zeros(
            (self.__num_points, 2 * self.__basis_number - 1), dtype=onp.complex64)
        self.__save_to_disk = save_to_disk
        self.__save_path = save_path
        self.__number_of_sample = number_of_sample
        self.__measurement = None
        self.__ground_truth_ready = False
        self.__measurement_matrix_ready = False
        self.__random_seed = random_seed
        onp.random.seed(self.__random_seed)

    @property
    def num_points(self) -> int:
        return self.__num_points

    @property
    def signal_shape(self) -> OneDSampleSignalShape:
        return self.__signal_shape

    @signal_shape.setter
    def signal_shape(self, value):
        if isinstance(value, OneDSampleSignalShape):
            self.__signal_shape = value
        else:
            raise ValueError('Signal shape must be an instance of OneDSampleSignalShape.')

    @property
    def measurement_matrix(self) -> onp.ndarray:
        if not self.__measurement_matrix_ready:
            for i in range(-self.__basis_number + 1, self.__basis_number):
                self.__measurement_matrix[:, i + self.__basis_number - 1] = _eigen_function_one_d(i, self.__t)

            self.__measurement_matrix_ready = True

        return self.__measurement_matrix

    @property
    def basis_number(self) -> int:
        return self.__basis_number

    @property
    def t(self):
        return self.__t

    @property
    def std_dev(self):
        return self.__std_dev

    def measure(self) -> onp.ndarray:
        if self.__number_of_sample == 1:
            meas_error = self.__std_dev * onp.random.randn(self.__t.shape[0])
        else:
            meas_error = self.__std_dev * onp.random.randn(self.__number_of_sample, self.__t.shape[0])
        self.__measurement = self.ground_truth + meas_error
        file_name = self.__signal_shape.name + '_N_sample_' + str(self.__number_of_sample) + '_stdev_' + str(
            self.__std_dev)
        save_directory = pathlib.Path(self.__save_path)
        if not save_directory.exists():
            save_directory.mkdir()
        file = save_directory / file_name
        if self.__save_to_disk:
            with h5py.File(file, 'w') as hdf_file:
                hdf_file.create_dataset('Measurement_samples', data=self.__measurement, compression='gzip')
                hdf_file.create_dataset('time', data=self.__t, compression='gzip')
                hdf_file.create_dataset('ground_truth', data=self.__ground_truth_signal, compression='gzip')
                hdf_file.create_dataset('stdev', data=self.__std_dev)
                hdf_file.create_dataset('random_seed', data=self.__random_seed)

        return self.__measurement

    @property
    def ground_truth(self):
        if not self.__ground_truth_ready:
            if self.__signal_shape == OneDSampleSignalShape.SMOOTH_DISCONTINUOUS:
                self.__ground_truth_signal = generate_smooth_discontinuous_signal(self.__t)

            elif self.__signal_shape == OneDSampleSignalShape.BOX:
                self.__ground_truth_signal = generate_box_signal(self.__t)

            elif self.__signal_shape == OneDSampleSignalShape.RECT:
                self.__ground_truth_signal = generate_rect_signal(self.__t)

            elif self.__signal_shape == OneDSampleSignalShape.RECT2:
                self.__ground_truth_signal = generate_rect_2_signal(self.__t)

            self.__ground_truth_ready = True

        if self.__ground_truth_ready:
            return self.__ground_truth_signal
        else:
            raise ValueError


def generate_smooth_discontinuous_signal(t: onp.ndarray):
    """
    Generate smooth discontinuous signal
    Parameters
    ----------
    t : onp.ndarray
        time array

    Returns
    -------
    out: onp.ndarray
        The ground truth signal
    """
    ground_truth_signal = onp.zeros_like(t)
    for i in range(t.shape[0]):
        if 0 < t[i] <= 0.5:
            ground_truth_signal[i] = onp.exp(4 - 1 / (2 * t[i] - 4 * t[i] ** 2))
            continue
        if 0.7 < t[i] <= 0.8:
            ground_truth_signal[i] = 1.
            continue
        if 0.8 < t[i] <= 0.9:
            ground_truth_signal[i] = -1.

    return ground_truth_signal


def generate_box_signal(t: onp.ndarray):
    """
    Generate box signal.
    Parameters
    ----------
    t : onp.ndarray
        time array

    Returns
    -------
    out: onp.ndarray
        The ground truth signal
    """
    ground_truth_signal = onp.zeros_like(t)
    ground_truth_signal[(0.2 < t) & (t < 0.8)] = 1.
    return ground_truth_signal


def generate_rect_signal(t: onp.ndarray):
    """
    Generate RECT signal.
    Parameters
    ----------
    t : onp.ndarray
        time array

    Returns
    -------
    out: onp.ndarray
        The ground truth signal
    """
    ground_truth_signal = onp.zeros_like(t)
    t1 = 1 / 7
    t2 = 2 / 7
    t3 = 3 / 7
    t4 = 4 / 7
    t5 = 5 / 7
    t6 = 6 / 7

    ground_truth_signal[(t >= 0) & (t < t1)] = 0
    ground_truth_signal[(t >= t1) & (t < t2)] = 1
    ground_truth_signal[(t >= t2) & (t < t3)] = 0
    ground_truth_signal[(t >= t3) & (t < t4)] = 0.6
    ground_truth_signal[(t >= t4) & (t < t5)] = 0
    ground_truth_signal[(t >= t5) & (t < t6)] = 0.4
    return ground_truth_signal


def generate_rect_2_signal(t: onp.ndarray):
    """
    Generate rect 2  signal.
    Parameters
    ----------
    t : onp.ndarray
        time array

    Returns
    -------
    out: onp.ndarray
        The ground truth signal
    """
    ground_truth_signal = onp.zeros_like(t)
    t1 = 1 / 4
    t2 = 2 / 4
    t3 = 3 / 4

    ground_truth_signal[(t >= 0) & (t < t1)] = 0
    ground_truth_signal[(t >= t1) & (t < t2)] = 1
    ground_truth_signal[(t >= t2) & (t < t3)] = 0.5

    return ground_truth_signal
