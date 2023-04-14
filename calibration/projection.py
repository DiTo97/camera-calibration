def compute_image_projection(points, K):
    """
    Compute projection of points onto the image plane

    Parameters
    -----------
    points - np.ndarray, shape - (3, n_points)
        points we want to project onto the image plane
        the points should be represented in the camera coordinate system
    K - np.ndarray, shape - (3, 3)
        camera intrinsic matrix

    Returns
    -------
    points_i - np.ndarray, shape - (2, n_points)
        the projected points on the image
    """

    h_points_i = K @ points

    h_points_i[0, :] = h_points_i[0, :] / h_points_i[2, :]
    h_points_i[1, :] = h_points_i[1, :] / h_points_i[2, :]

    points_i = h_points_i[:2, :]

    return points_i


def compute_coordniates_wrt_camera(world_points, E, is_homogeneous=False):
    """
    Performs a change of basis operation from the world coordinate system
    to the camera coordinate system

    Parameters
    ------------
    world_points - np.ndarray, shape - (3, n_points) or (4, n_points)
             points in the world coordinate system
    E - np.ndarray, shape - (3, 4)
        the camera extrinsic matrix
    is_homogeneous - boolean
        whether the coordinates are represented in their homogeneous form
        if False, an extra dimension will  be added for computation

    Returns
    ----------
    points_c - np.ndarray, shape - (3, n_points)
             points in the camera coordinate system
    """
    if not is_homogeneous:
        # convert to homogeneous coordinates
        points_h = np.vstack((world_points, np.ones(world_points.shape[1])))

    points_c = E @ points_h
    return points_c


def create_algebraic_matrix(world_points, projections):
    """
    Create the algebraic matrix A for camera calibration

    Parameters
    -----------
    world points - np.ndarray, shape - (3, n_points)
        points in the world coordinate system

    projections - np.ndarray, shape - (3, n_points)
        projections of the above points in the image

    Returns
    ----------
    A - np.ndarray, shape - (2 * n_points, 12)
        the algebraic matrix used for camera calibration
    """

    assert world_points.shape[1] == projections.shape[1]
    n_points = world_points.shape[1]
    A = np.ones(shape=(2 * n_points, 12))

    c = 0

    for i in range(n_points):

        w = world_points[:, i]
        p = projections[:, i]

        X, Y, Z = w
        u, v = p
        rows = np.zeros(shape=(2, 12))

        rows[0, 0], rows[0, 1], rows[0, 2], rows[0, 3] = X, Y, Z, 1
        rows[0, 8], rows[0, 9], rows[0, 10], rows[0, 11] = -u * X, -u * Y, -u * Z, -u

        rows[1, 4], rows[1, 5], rows[1, 6], rows[1, 7] = X, Y, Z, 1
        rows[1, 8], rows[1, 9], rows[1, 10], rows[1, 11] = -v * X, -v * Y, -v * Z, -v

        A[c : c + 2, :] = rows
        c += 2

    return A


def compute_world2img_projection(world_points, M, is_homogeneous=False):
    """
    Given a set of points in the world and the overall camera matrix,
    compute the projection of world points onto the image

    Parameters
    -----------
    world_points - np.ndarray, shape - (3, n_points)
                   points in the world coordinate system

    M - np.ndarray, shape - (3, 4)
        The overall camera matrix which is a composition of the extrinsic and intrinsic matrix

    is_homogeneous - boolean
        whether the coordinates are represented in their homogeneous form
        if False, an extra dimension will  be added for computation

    Returns
    ----------
    projections - np.ndarray, shape - (2, n_points)
                  projections of the world points onto the image
    """
    if not is_homogeneous:
        # convert to homogeneous coordinates
        points_h = np.vstack((world_points, np.ones(world_points.shape[1])))

    h_points_i = M @ points_h

    h_points_i[0, :] = h_points_i[0, :] / h_points_i[2, :]
    h_points_i[1, :] = h_points_i[1, :] / h_points_i[2, :]

    points_i = h_points_i[:2, :]

    return points_i
