import numpy as np
import numpy.typing as nptyping


def intrinsic_matrix(
    f: float, s: float, a: float, cx: float, cy: float
) -> nptyping.NDArray[np.float_]:
    """It generates an intrinsic camera matrix with the given args

    Parameters
    ----------
    f
        The focal length that converts pixel units to world units (e.g. mm)
    s
        The skew between the x-axis and the y-axis
    a
        The aspect ratio
    cx
        The principal point x-coordinate
    cy
        The principal point y-coordinate

    Returns
    -------
    K
        The 3-by-3 intrinsic camera matrix
    """
    K = np.identity(3)

    K[0, 0] = f
    K[0, 1] = s
    K[1, 1] = f * a
    K[0, 2] = cx
    K[1, 2] = cy

    return K


def extrinsic_matrix(
    R: nptyping.NDArray[np.float_], t: nptyping.NDArray[np.float_]
) -> nptyping.NDArray[np.float_]:
    """It generates an extrinsic camera matrix with the given args

    Parameters
    ----------
    R
        A 3-by-3 rotation matrix
    t
        A 3-by-1 translation vector

    Returns
    -------
    E
        The 3-by-4 extrinsic camera matrix
    """
    E = np.identity(4)

    E[:3, :3] = R
    E[:3, -1] = t

    E = np.linalg.inv(E)  # The change of basis matrix
    E = E[:-1, :]

    return E
