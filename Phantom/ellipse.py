import numpy as onp
import jax.numpy as jnp
from collections import namedtuple
from typing import List

# The angle of ellipse given in rad
Ellipse = namedtuple("Ellipse", ["x_center", "y_center", "major_axis", "minor_axis", "angle", "gray_level"])


def multi_ellipses_fun(x_points: onp.ndarray, y_points: onp.ndarray, ellipses: List[Ellipse]) -> onp.ndarray:
    """
    Evaluate a set of points in `x_points`,`y_points`, such that if the point is inside the ellipses, then it will
    outputs the summation of the ellipses gray level it inside, otherwise return zero.

    Parameters
    ----------
    x_points: numpy.ndarray
    y_points: numpy.ndarray
    ellipses: List[Ellipse]

    Returns
    -------
    out: numpy.array
        An array of the same size with x_points and y_points
    """

    output = onp.zeros(x_points.shape)
    for ellipse in ellipses:
        output += ellipse_fun(x_points, y_points, ellipse)
    return output


def ellipse_fun(x_points: onp.ndarray, y_points: onp.ndarray, ellipse: Ellipse) -> onp.ndarray:
    """
    Evaluate a set of points in `x_points`,`y_points`, such that if the point is inside the ellipse, then it will
    outputs the ellipse gray level, otherwise return zero.

    Parameters
    ----------
    x_points: numpy.ndarray
    y_points: numpy.ndarray
    ellipse: Ellipse

    Returns
    -------
    out: numpy.array
        An array of the same size with x_points and y_points
    """
    output = onp.zeros(x_points.shape)
    sin_theta = onp.sin(ellipse.angle)
    cos_theta = onp.cos(ellipse.angle)
    delta_x = (x_points - ellipse.x_center)
    delta_y = (y_points - ellipse.y_center)
    rotated_delta_x = (delta_x * cos_theta - delta_y * sin_theta)
    rotated_delta_y = (delta_x * sin_theta + delta_y * cos_theta)

    mask = ((rotated_delta_x / ellipse.major_axis) ** 2 + (rotated_delta_y / ellipse.minor_axis) ** 2) < 1.
    output[mask] = ellipse.gray_level
    return output


def transform_ellipses(x_shift: onp.ndarray, y_shift: onp.ndarray, major_axis_scale: onp.ndarray,
                       minor_axis_scale: onp.ndarray, angle_shift: onp.ndarray, gray_scale: onp.ndarray,
                       ellipses: List[Ellipse]):
    """
    Transform uniformly collection of ellipses, by specifying the the two dimensional shifting (`x_shift`, `y_shift`),
    and scaling on both major and minor axis (`major_axis_scale`,`minor_axis_scale`) and angle shift `angle_shift`.


    Parameters
    ----------
    x_shift: numpy.ndarray Lx1
    y_shift: numpy.ndarray Lx1
    major_axis_scale: numpy.ndarray Lx1
    minor_axis_scale: numpy.ndarray Lx1
    angle_shift: numpy.ndarray Lx1
    gray_scale: numpy.ndarray Lx1
    ellipses: List[Ellipse]

    Returns
    -------
    out: List[Ellipse]
        The transformed ellipse
    """
    new_ellipses = []
    for ellipse, i in zip(ellipses, range(len(ellipses))):
        new_ellipse = Ellipse(ellipse.x_center + x_shift[i],
                              ellipse.y_center + y_shift[i],
                              ellipse.major_axis * major_axis_scale[i],
                              ellipse.minor_axis * minor_axis_scale[i],
                              ellipse.angle + angle_shift[i],
                              ellipse.gray_level * gray_scale[i]
                              )
        new_ellipses.append(new_ellipse)

    return new_ellipses


def scale_ellipses(positional_scale: float, gray_scale: float,
                   ellipses: List[Ellipse]):
    """
    Scale ellipses uniformly.

    Parameters
    ----------
    positional_scale: float
        positional scale.
    gray_scale: float
        gray_scaling.
    ellipses: List[Ellipse]
        the ellipses.

    Returns
    -------
    out:List[Ellipse]
        scaled elllipses.
    """
    new_ellipses = []
    for ellipse, i in zip(ellipses, range(len(ellipses))):
        new_ellipse = Ellipse(ellipse.x_center * positional_scale,
                              ellipse.y_center * positional_scale,
                              ellipse.major_axis * positional_scale,
                              ellipse.minor_axis * positional_scale,
                              ellipse.angle,
                              ellipse.gray_level * gray_scale
                              )
        new_ellipses.append(new_ellipse)

    return new_ellipses


def compute_distance(x_c: float, y_c: float, a: float, b: float, phi: float, p_prime: onp.ndarray, theta: onp.ndarray) \
        -> onp.ndarray:
    """
    Compute a distance between two points of intersection between line that is perpendicular to the line with angle
    theta passing the origin, at the point p_prime, and an ellipse with center at `x_c`, `y_c`, with major and minor
    axis (`a`,`b`), with angle `phi`.

    Parameters
    ----------
    x_c: float
        X axis of the center of the ellipse.
    y_c: float
        Y axis of the center of the ellipse.
    a: float
        Major axis of the ellipse.
    b: float
        Minor axis of the ellipse.
    phi: float
        Angle axis of the ellipse.
    p_prime: numpy.ndarray
        Range of the integration.
    theta: numpy.ndarray
        Projection angle.

    Returns
    -------
    out: numpy.ndarray
        distance.
    """
    c_theta = onp.cos(theta)
    s_theta = onp.sin(theta)
    c_square_theta_min_phi = onp.cos(theta - phi) ** 2
    s_square_theta_min_phi = onp.sin(theta - phi) ** 2
    ones_p = onp.ones(p_prime.shape[0])
    ones_t = onp.ones(theta.shape[0])
    d = (-onp.outer(p_prime, ones_t) + x_c * onp.outer(ones_p, c_theta) - y_c * onp.outer(ones_p, s_theta))
    d_square = d * d
    a_square = a * a
    b_square = b * b
    factor_1 = (a_square * b_square * c_square_theta_min_phi * c_square_theta_min_phi) + \
               (b_square * b_square * s_square_theta_min_phi * s_square_theta_min_phi) - \
               (b_square * d_square) + (
                           b_square * (a_square + b_square) * s_square_theta_min_phi * c_square_theta_min_phi)

    factor_2 = (a_square * c_square_theta_min_phi + b_square * s_square_theta_min_phi)

    distance = 2 * a * onp.sqrt(factor_1) / factor_2
    return onp.nan_to_num(distance)


def sinogram_of_ellipses(ellipses: List[Ellipse], p_prime: jnp.ndarray, theta: jnp.ndarray) \
        -> onp.ndarray:
    """
    Compute sinogram of a collection of Ellipse

    Parameters
    ----------
    ellipses: List[Ellipse]
        The ellipses
    p_prime: numpy.array
        P_prime
    theta: numpy.array
        Projection angles.

    Returns
    -------
    out: numpy.array
        Tshe sinogram of the ellipses.
    """
    distance = onp.zeros((p_prime.shape[0], theta.shape[0]))
    for ellipse in ellipses:
        distance += ellipse.gray_level * compute_distance(ellipse.x_center, ellipse.y_center, ellipse.major_axis,
                                                          ellipse.minor_axis,
                                                          ellipse.angle, p_prime, theta)
    return distance


"""
Based on the description in
Shepp, Larry A.; Logan, Benjamin F. (June 1974). 
"The Fourier Reconstruction of a Head Section". 
IEEE Transactions on Nuclear Science
"""
original_shepp_logan_ellipses = [Ellipse(0., 0., 0.69, 0.92, 0., 2.),
                                 Ellipse(0., -0.0184, 0.6624, 0.874, 0., -0.98),
                                 Ellipse(0.22, 0., 0.11, 0.31, 0.1 * onp.pi, -0.02),
                                 Ellipse(-0.22, 0., 0.16, 0.41, -0.1 * onp.pi, -0.02),
                                 Ellipse(0., 0.35, 0.21, 0.25, 0., 0.01),
                                 Ellipse(0., 0.1, 0.046, 0.046, 0., 0.01),
                                 Ellipse(0., -0.1, 0.046, 0.046, 0., 0.01),
                                 Ellipse(-0.08, -0.605, 0.046, 0.023, 0., 0.01),
                                 Ellipse(0., -0.605, 0.023, 0.023, 0., 0.01),
                                 Ellipse(0.06, -0.605, 0.023, 0.023, 0., 0.01)]

__zeros = onp.zeros(len(original_shepp_logan_ellipses))
__ones = onp.ones(len(original_shepp_logan_ellipses))
__gray_scale = onp.ones(len(original_shepp_logan_ellipses))
__gray_scale[2:] = 40
__scaled_shepp_logan = scale_ellipses(0.5, 1., original_shepp_logan_ellipses)

contrast_shepp_logan_ellipses = transform_ellipses(__zeros, __zeros, __ones, __ones, __zeros, __gray_scale,
                                                   __scaled_shepp_logan)

single_ellipse = [Ellipse(0.25, 0.25, 0.05, 0.05, 0., 2.)]

