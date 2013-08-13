# coding: utf-8

from numpy import concatenate, rollaxis, indices
from numpy.random import randint as np_randint
from numpy cimport ndarray, float64_t
from scipy.spatial.ckdtree import cKDTree
from utils cimport save_to_img


cpdef ndarray[double, ndim=2] voronoi_array(
        int size, int n_points=15, bint save=False):
    cdef ndarray[int, ndim=2] points = np_randint(
        0, size, (n_points, 2)).astype(b'int32')
    tree = cKDTree(concatenate([
        points,
        points - [size, 0], points + [size, 0],
        points - [0, size], points + [0, size],
        points - [size, size], points + [size, size],
        points + [-size, size], points + [size, -size]]))

    # Taken from http://stackoverflow.com/a/4714857/1576438
    cdef ndarray[int, ndim=2] a = rollaxis(
        indices([size, size], dtype=b'int32'), 0, 3).reshape(-1, 2)
    cdef ndarray[double, ndim=2] m = tree.query(a, 2)[0]
    m = (-m[..., 0] + m[..., 1]).reshape(size, size)

    if save:
        save_to_img(m)

    return m
