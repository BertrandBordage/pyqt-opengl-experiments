# coding: utf-8

from __future__ import division
from libc.stdlib cimport rand, RAND_MAX
from numpy import concatenate, rollaxis, indices, array
from numpy.random import randint as np_randint
from numpy cimport ndarray
from scipy.spatial.ckdtree import cKDTree
from utils cimport save_to_img


cdef inline double random() nogil:
    return rand() / RAND_MAX


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
    cdef ndarray[double, ndim=1] points_heights = array([
        random() if random() < 0.5 else 0 for _ in range(n_points)] * 9)

    # Taken from http://stackoverflow.com/a/4714857/1576438
    cdef ndarray[int, ndim=2] a = rollaxis(
        indices([size, size], dtype=b'int32'), 0, 3).reshape(-1, 2)
    cdef tuple query = tree.query(a, 2)
    cdef ndarray[double, ndim=2] m = query[0].swapaxes(0, -1)
    cdef ndarray[long, ndim=2] height_indices = query[1]
    cdef ndarray[double, ndim=1] heights = points_heights[height_indices[..., 0]]
    m = ((m[1] - m[0]) * heights).reshape(size, size)

    if save:
        save_to_img(m)

    return m
