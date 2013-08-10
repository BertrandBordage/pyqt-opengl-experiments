# coding: utf-8

# cython: boundscheck=False
# cython: cdivision=True
# cython: c_string_type=bytes

from __future__ import unicode_literals, division
import numpy as np
cimport numpy as np
from PIL import Image
from utils cimport uniform


i = 0


cdef inline void save_to_img(np.ndarray m):
    global i
    cdef float mini = m.min()
    if mini < 0:
        m += - mini
    m *= 255 / m.max()

    img = Image.fromarray(m.astype(b'uint8'))
    img.save('diamond_square/map%s.png' % i)
    i += 1


cdef inline int _get_real_min(int mini, int size) nogil:
    if -size < mini < 0:
        return mini - 1
    return mini


cdef inline int _get_real_max(int maxi, int size) nogil:
    if maxi >= size:
        return maxi - (size - 1)
    return maxi


cpdef np.ndarray[double, ndim=2] build_height_map(
        int size, int amplitude=15, int smoothing=10, bint save=False):
    cdef int orig_size, step, x, y, xmin, xmax, ymin, ymax
    cdef float random_coef
    orig_size = size
    size += (size + 1) % 2

    cdef np.ndarray[double, ndim=2] m = np.zeros((size, size))

    step = size // 2
    while True:
        random_coef = <double>step / <double>smoothing

        # Diamond
        for x from step <= x < size by step * 2:
            for y from step <= y < size by step * 2:
                xmin = _get_real_min(x - step, size)
                ymin = _get_real_min(y - step, size)
                xmax = _get_real_max(x + step, size)
                ymax = _get_real_max(y + step, size)
                m[x, y] = (m[xmin, ymin] + m[xmax, ymin]
                           + m[xmin, ymax] + m[xmax, ymax]
                           + uniform(-random_coef, random_coef)) / 4

        # Square
        for x from 0 <= x < size by step:
            for y from 0 <= y < size by step:
                if m[x, y]:
                    continue

                xmin = _get_real_min(x - step, size)
                ymin = _get_real_min(y - step, size)
                xmax = _get_real_max(x + step, size)
                ymax = _get_real_max(y + step, size)
                m[x, y] = (m[x, ymin] + m[xmin, y]
                           + m[xmax, y] + m[x, ymax]) / 4

        if step == 1:
            break
        step //= 2

    m = m[:orig_size, :orig_size] * amplitude

    if save:
        save_to_img(m.copy())

    return m
