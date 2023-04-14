import typing

import numpy as np
import numpy.typing as nptyping


def random_points_3d(
    num_points: int,
    xlim: typing.Tuple[int, int],
    ylim: typing.Tuple[int, int],
    zlim: typing.Tuple[int, int],
) -> nptyping.NDArray[np.int_]:
    """It generates 3-dimensional random points in the given limits"""
    x = np.random.randint(xlim[0], xlim[1], size=num_points)
    y = np.random.randint(ylim[0], ylim[1], size=num_points)
    z = np.random.randint(zlim[0], zlim[1], size=num_points)

    return np.vstack((x, y, z))
