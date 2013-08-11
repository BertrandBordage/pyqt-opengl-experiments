cimport numpy as np

ctypedef fused real:
    short
    int
    long
    float
    double

cdef inline real uniform(real a, real b) nogil

cpdef inline save_to_img(np.ndarray m)
