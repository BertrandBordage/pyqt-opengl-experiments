# coding: utf-8

import numpy as np
cimport numpy as np
from utils cimport save_to_img


cdef np.ndarray[double, ndim=1] distances_from_array(np.ndarray[double, ndim=2] array, int x, int y):
    cdef np.ndarray[double, ndim=2] d = (array - [x, y]) ** 2
    return d[:, 0] + d[:, 1]


cpdef np.ndarray[double, ndim=2] voronoi_array(
        int size, int n_points=15, bint save=True):
    cdef np.ndarray[double, ndim=2] points = np.random.randint(
        0, size, (n_points, 2)).astype('double')
    cdef np.ndarray[double, ndim=2] vertices = np.concatenate(
        [points,
         points - [size, 0], points + [size, 0],
         points - [0, size], points + [0, size],
         points - [size, size], points + [size, size],
         points + [-size, size], points + [size, -size]])

    cdef np.ndarray[double, ndim=2] m = np.zeros((size, size), dtype=b'double')
    cdef np.ndarray[double, ndim=1] distances
    cdef int x, y
    for x in range(m.shape[0]):
        for y in range(m.shape[1]):
            distances = distances_from_array(vertices, x, y)
            distances.sort()
            m[x, y] = -distances[0] + distances[1]

    if save:
        save_to_img(m)

    # We should return np.sqrt(m) but to optimize, we skip this step.
    return m
