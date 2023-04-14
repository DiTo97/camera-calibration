import typing

import numpy as np
import numpy.typing as nptyping


def rotation_matrix_x(angle: float) -> nptyping.NDArray[np.float_]:
    """The transformation matrix that rotates a point about the x-axis

    Parameters
    ----------
    angle
        The rotation angle in radians

    Returns
    -------
        The 3-by-3 rotation matrix
    """
    R_x = np.zeros(shape=(3, 3))

    R_x[0, 0] = 1
    R_x[1, 1] = np.cos(angle)
    R_x[1, 2] = -np.sin(angle)
    R_x[2, 1] = np.sin(angle)
    R_x[2, 2] = np.cos(angle)

    return R_x


def rotation_matrix_y(angle: float) -> nptyping.NDArray[np.float_]:
    """The transformation matrix that rotates a point about the y-axis

    Parameters
    ----------
    angle
        The rotation angle in radians

    Returns
    -------
        The 3-by-3 rotation matrix
    """
    R_y = np.zeros(shape=(3, 3))

    R_y[0, 0] = np.cos(angle)
    R_y[0, 2] = -np.sin(angle)
    R_y[1, 1] = 1
    R_y[2, 0] = np.sin(angle)
    R_y[2, 2] = np.cos(angle)

    return R_y


def rotation_matrix_z(angle: float) -> nptyping.NDArray[np.float_]:
    """The transformation matrix that rotates a point about the z-axis

    Parameters
    ----------
    angle
        The rotation angle in radians

    Returns
    -------
        The 3-by-3 rotation matrix
    """
    R_z = np.zeros(shape=(3, 3))

    R_z[0, 0] = np.cos(angle)
    R_z[0, 1] = -np.sin(angle)
    R_z[1, 0] = np.sin(angle)
    R_z[1, 1] = np.cos(angle)
    R_z[2, 2] = 1

    return R_z


def rotation_matrix(
    angles: typing.List[float], order: str
) -> nptyping.NDArray[np.float_]:
    """The transformation matrix that rotates a vector through the given angles

    The rotation is carried out in the given order w.r.t. the standard basis axial system

    Parameters
    -----------
    angles
        The rotation angles in radians
    order
        The rotation order

    Returns
    --------
    R
        The 3-by-3 rotation matrix

    Notes
    -----
        The rotation is carried out anti-clockwise in a left-handed axial system
    """
    mapping = {"x": rotation_matrix_x, "y": rotation_matrix_y, "z": rotation_matrix_z}

    R = np.identity(3)

    for angle, axis in list(zip(angles, order))[::-1]:
        transform = mapping.get(axis)

        if transform is None:
            message = f"Invalid axis: {axis}"
            raise ValueError(message)

        axial = transform(angle)
        R = np.matmul(R, axial)

    return R
