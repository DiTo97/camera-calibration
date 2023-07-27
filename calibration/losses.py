import numpy as np
import numpy.typing as np_typing


Transform1d = np_typing.NDArray[np.float_]
"""A 12-dim transform vector"""

Transform2d = np_typing.NDArray[np.float_]
"""A 3-by-4 transform matrix"""

Cloud2d = np_typing.NDArray[np.float_]
"""A heterogeneous 2-dim point cloud"""

Cloud3d = np_typing.NDArray[np.float_]
"""A heterogeneous 3-dim point cloud"""


def geometric_error(m: Transform1d, world_points: Cloud3d, image_points: Cloud2d) -> float:
    """The geometric error between image points and world points' projections

    The world points are projected with the given transform.

    Parameters
    ----------
    m
        The 12-dim transform vector

    world_points
        The 3-by-n point cloud in world coordinate system

    image_points
        The 2-by-n point cloud in image coordinate system

    Returns
    -------
    error
        The geometric error
    """
    num_points = world_points.shape[1]

    assert num_points == image_points.shape[1]

    M = m.reshape((3, 4))

    world_points = np.vstack((world_points, np.ones(num_points)))  # homogeneous

    projections_2d = M @ world_points

    projections_2d[0, :] /= projections_2d[2, :]
    projections_2d[1, :] /= projections_2d[2, :]

    projections_2d = projections_2d[:2, :]  # heterogeneous

    residual = projections_2d - points_2d
    residual = np.sqrt(np.sum(np.square(residual), axis=0))

    error = residual.sum()

    return error
