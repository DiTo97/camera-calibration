def geometric_error(m, world_points, projections):
    """
    compute the geometric error wrt the
    prediction projections and the groundtruth projections

    Parameters
    ------------
    m - np.ndarray, shape - (12)
        an 12-dim vector which is to be updated
    world_points - np.ndarray, shape - (3, n)
                   points in the world coordinate system
    projections - np.ndarray(2, n)
                  projections of the points in the image

    Returns
    --------
    error - float
            the geometric error
    """
    assert world_points.shape[1] == projections.shape[1]
    error = 0
    n_points = world_points.shape[1]
    for i in range(n_points):
        X, Y, Z = world_points[:, i]
        u, v = projections[:, i]
        u_ = m[0] * X + m[1] * Y + m[2] * Z + m[3]
        v_ = m[4] * X + m[5] * Y + m[6] * Z + m[7]
        d = m[8] * X + m[9] * Y + m[10] * Z + m[11]
        u_ = u_ / d
        v_ = v_ / d
        error += np.sqrt(np.square(u - u_) + np.square(v - v_))
    return error
