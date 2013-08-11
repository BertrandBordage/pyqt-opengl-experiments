cimport numpy as np

cpdef np.ndarray[double, ndim=2] build_height_map(
        int size, int amplitude=*, int smoothing=*, bint save=*)
