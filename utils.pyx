# coding: utf-8

# cython: boundscheck=False
# cython: cdivision=True


from libc.stdlib cimport rand, RAND_MAX


cdef inline float uniform(float a, float b):
    return a + (b - a) * rand() / RAND_MAX
