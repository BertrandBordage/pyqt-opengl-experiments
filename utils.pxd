ctypedef fused real:
    short
    int
    long
    float
    double

cdef inline real uniform(real a, real b) nogil
