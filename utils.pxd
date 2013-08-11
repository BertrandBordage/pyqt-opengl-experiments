cimport numpy as np

ctypedef fused real:
    short
    int
    long
    float
    double

cdef inline real uniform(real a, real b) nogil

cpdef inline np.ndarray[double, ndim=2] equalize_height_map(
        np.ndarray[double, ndim=2] hmap, double m, double M)

cpdef inline save_to_img(np.ndarray m)
