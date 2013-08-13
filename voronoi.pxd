from numpy cimport ndarray

cpdef ndarray[double, ndim=2] voronoi_array(
        int size, int n_points=?, bint save=?)
