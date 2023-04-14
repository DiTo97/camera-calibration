import numpy as np


make_line = lambda u, v: np.vstack((u, v)).T


def create_image_grid(f, img_size):
    """
    Create an image grid of the given size parallel to the XY plane
    at a distance f from the camera center (origin)
    """
    h, w = img_size
    xx, yy = np.meshgrid(range(-(h // 2), w // 2 + 1), range(-(h // 2), w // 2 + 1))
    Z = np.ones(shape=img_size) * f

    return xx, yy, Z


def convert_grid_to_homogeneous(xx, yy, Z, img_size):
    """
    Extract coordinates from a grid and convert them to homogeneous coordinates
    """
    h, w = img_size
    pi = np.ones(shape=(4, h * w))
    c = 0
    for i in range(h):
        for j in range(w):
            x = xx[i, j]
            y = yy[i, j]
            z = Z[i, j]
            point = np.array([x, y, z])
            pi[:3, c] = point
            c += 1
    return pi


def convert_homogeneous_to_grid(pts, img_size):
    """
    Convert a set of homogeneous points to a grid
    """
    xxt = pts[0, :].reshape(img_size)
    yyt = pts[1, :].reshape(img_size)
    Zt = pts[2, :].reshape(img_size)

    return xxt, yyt, Zt
