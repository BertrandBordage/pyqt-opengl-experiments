cimport numpy as np

cpdef np.ndarray[double, ndim=2] build_height_map(
    int size, int amplitude=*, float smoothing=*, bint save=*)
