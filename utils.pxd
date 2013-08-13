from numpy cimport ndarray

ctypedef fused real:
    short
    int
    long
    float
    double

cdef inline real uniform(real a, real b) nogil

cpdef inline ndarray[double, ndim=2] equalize_height_map(
        ndarray[double, ndim=2] hmap, double m, double M)

cpdef inline save_to_img(ndarray m)
