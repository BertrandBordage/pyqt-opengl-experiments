# coding: utf-8

# cython: boundscheck=False
# cython: cdivision=True
# cython: c_string_type=bytes

from __future__ import unicode_literals, division
from libc.math cimport pow
import numpy as np
cimport numpy as np
from utils cimport uniform, save_to_img


cdef inline int _get_real_min(int mini, int size) nogil:
    if -size < mini < 0:
        return mini - 1
    return mini


cdef inline int _get_real_max(int maxi, int size) nogil:
    if maxi >= size:
        return maxi - (size - 1)
    return maxi


cdef enum SIDE:
    DIAMOND = 1
    SQUARE = 2


cdef inline double _nearby_sum(double[:, :] m, int x, int y,
                               int step, int size, SIDE side) nogil:
    cdef int xm, xM, ym, yM
    xm = _get_real_min(x - step, size)
    xM = _get_real_max(x + step, size)
    ym = _get_real_min(y - step, size)
    yM = _get_real_max(y + step, size)

    if side == DIAMOND:
        return m[xm, ym] + m[xM, ym] + m[xm, yM] + m[xM, yM]
    return m[x, ym] + m[xm, y] + m[xM, y] + m[x, yM]


cdef inline void _update_borders(double[:, :] m_view, int x, int y) nogil:
    if x == 0:
        m_view[-1, y] = m_view[0, y]
    elif y == 0:
        m_view[x, -1] = m_view[x, 0]


cdef inline float _random_coef(unsigned int step, float smoothing) nogil:
    return pow(<float>step, smoothing)


cpdef np.ndarray[double, ndim=2] continuous_map(
        int size, int amplitude=15, float smoothing=0.8, bint save=True):
    cdef unsigned int orig_size, step, two_steps, x, y
    orig_size = size
    size += (size + 1) % 2

    cdef np.ndarray[double, ndim=2] m = np.zeros((size, size))
    cdef double[:, :] m_view = m


    step = size // 2
    cdef float random_coef = _random_coef(step, smoothing)
    m_view[0, 0] = m_view[-1, 0] = m_view[0, -1] = m_view[-1, -1] = random_coef

    while True:
        random_coef = _random_coef(step, smoothing)
        two_steps = step * 2

        # Diamond
        for x from step <= x < size by two_steps:
            for y from step <= y < size by two_steps:
                if m_view[x, y]:
                    continue

                m_view[x, y] = (_nearby_sum(m_view, x, y, step, size, DIAMOND)
                                + uniform(-random_coef, random_coef)) / 4

                _update_borders(m_view, x, y)

        # Square
        for x from 0 <= x < size by step:
            for y from 0 <= y < size by step:
                if m_view[x, y]:
                    continue

                m_view[x, y] = _nearby_sum(m_view, x, y, step, size, SQUARE) / 4

                _update_borders(m_view, x, y)

        if step == 1:
            break
        step //= 2

    m = m[:orig_size, :orig_size] * amplitude

    if save:
        save_to_img(m)

    return m
