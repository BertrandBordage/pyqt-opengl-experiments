from numpy cimport ndarray

cpdef ndarray[double, ndim=2] perturbate_array(
        ndarray[double, ndim=2] height_map, bint save=?)
