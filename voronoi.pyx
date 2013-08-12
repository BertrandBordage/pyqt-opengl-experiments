# coding: utf-8

import numpy as np
cimport numpy as np
from scipy.spatial import cKDTree
from utils cimport save_to_img


cpdef np.ndarray[double, ndim=2] voronoi_array(
        int size, int n_points=15, bint save=True):
    cdef np.ndarray[double, ndim=2] points = np.random.randint(
        0, size, (n_points, 2)).astype('double')
    tree = cKDTree(np.concatenate([
        points,
        points - [size, 0], points + [size, 0],
        points - [0, size], points + [0, size],
        points - [size, size], points + [size, size],
        points + [-size, size], points + [size, -size]]))

    cdef int x, y
    cdef list a = []
    for x in range(size):
        for y in range(size):
            a.append([x, y])
    cdef np.ndarray[double, ndim=2] m = tree.query(a, 2)[0]
    m = (-m[:, 0] + m[:, 1]).reshape(size, size)

    if save:
        save_to_img(m)

    return m
