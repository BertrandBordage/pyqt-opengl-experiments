from numpy cimport ndarray

cpdef ndarray[double, ndim=2] continuous_map(
    int size, int amplitude=?, float smoothing=?, bint save=?)
