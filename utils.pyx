# coding: utf-8

# cython: boundscheck=False
# cython: cdivision=True


from libc.stdlib cimport rand, RAND_MAX
cimport numpy as np
import os
from PIL import Image


cdef inline real uniform(real a, real b) nogil:
    return a + (b - a) * rand() / RAND_MAX


i = 0


cpdef np.ndarray[double, ndim=2] equalize_height_map(
        np.ndarray[double, ndim=2] hmap, double m, double M):
    hmap -= hmap.min()
    cdef double hmap_max = hmap.max()
    if hmap_max == 0.0:
        return hmap
    return m + (M - m) * hmap / hmap_max


cpdef inline save_to_img(np.ndarray m):
    global i
    m = equalize_height_map(m, 0.0, 255.0)

    directory = 'maps'

    if not os.path.exists(directory):
        os.makedirs(directory)

    img = Image.fromarray(m.astype(b'uint8'))
    img.save(os.path.join(directory, 'map%s.png' % i))
    i += 1
