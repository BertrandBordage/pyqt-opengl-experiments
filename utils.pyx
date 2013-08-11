# coding: utf-8

# cython: boundscheck=False
# cython: cdivision=True


from libc.stdlib cimport rand, RAND_MAX
cimport cython
cimport numpy as np
from PIL import Image


cdef inline real uniform(real a, real b) nogil:
    return a + (b - a) * rand() / RAND_MAX


i = 0


cpdef inline save_to_img(np.ndarray m):
    global i
    cdef float mini = m.min()
    if mini < 0:
        m += - mini
    m *= 255 / m.max()

    img = Image.fromarray(m.astype(b'uint8'))
    img.save('diamond_square/map%s.png' % i)
    i += 1
