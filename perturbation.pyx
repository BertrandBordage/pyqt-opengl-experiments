# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport cos, sin, M_PI
from numpy cimport import_array, ndarray, PyArray_EMPTY, NPY_DOUBLE
from diamond_square cimport continuous_map
from utils cimport equalize_height_map, save_to_img


import_array()


cpdef ndarray[double, ndim=2] perturbate_array(
        ndarray[double, ndim=2] height_map, bint save=False):
    cdef int size = height_map.shape[0]
    DEF magnitude = 0.0625
    cdef ndarray[double, ndim=2] angles = \
        equalize_height_map(continuous_map(size), -M_PI, M_PI)
    cdef ndarray[double, ndim=2] distances = \
        equalize_height_map(continuous_map(size), 0.0, size * magnitude)
    cdef ndarray[double, ndim=2] new_height_map = \
        PyArray_EMPTY(2, [size, size], NPY_DOUBLE, 0)
    cdef int x, y, new_x, new_y
    cdef float a, d
    for x in range(height_map.shape[0]):
        for y in range(height_map.shape[1]):
            a = angles[x, y]
            d = distances[x, y]
            new_x = x + <int>(d * cos(a))
            new_y = y + <int>(d * sin(a))
            if new_x < 0:
                new_x += size
            elif new_x >= size:
                new_x -= size
            if new_y < 0:
                new_y += size
            elif new_y >= size:
                new_y -= size
            new_height_map[x, y] = height_map[new_x, new_y]

    if save:
        save_to_img(new_height_map)

    return new_height_map
