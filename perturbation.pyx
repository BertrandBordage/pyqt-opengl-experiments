from libc.math cimport cos, sin, M_PI
cimport numpy as np
from diamond_square cimport continuous_map
from utils cimport equalize_height_map, save_to_img


cpdef np.ndarray[double, ndim=2] perturbate_array(
        np.ndarray[double, ndim=2] height_map, bint save=True):
    cdef int size = height_map.shape[0]
    DEF magnitude = 0.25
    cdef np.ndarray[double, ndim=2] angles, distances, new_height_map
    angles = equalize_height_map(continuous_map(size), -M_PI, M_PI)
    distances = equalize_height_map(continuous_map(size), 0.0, size * magnitude)
    new_height_map = height_map.copy()
    cdef int x, y, new_x, new_y
    cdef float a, d
    for x in range(height_map.shape[0]):
        for y in range(height_map.shape[1]):
            a = angles[x, y]
            d = distances[x, y]
            new_x = <int> (x + d * cos(a))
            new_y = <int> (y + d * sin(a))
            if new_x >= size:
                new_x -= size
            elif new_x < 0:
                new_x += size
            if new_y >= size:
                new_y -= size
            elif new_y < 0:
                new_y += size
            new_height_map[x, y] = height_map[new_x, new_y]

    if save:
        save_to_img(new_height_map)

    return new_height_map
