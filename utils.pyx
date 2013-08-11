# coding: utf-8

# cython: boundscheck=False
# cython: cdivision=True


from libc.stdlib cimport rand, RAND_MAX
cimport cython


cdef inline real uniform(real a, real b) nogil:
    return a + (b - a) * rand() / RAND_MAX
