import numpy as np
cimport numpy as np
cimport cython


ctypedef np.complex_t CTYPE_t
ctypedef double complex DCTYPE_t
ctypedef np.float_t FTYPE_t

cpdef np.ndarray[DCTYPE_t, ndim=2] iterativeBuildUpRinv(np.ndarray[DCTYPE_t, ndim=2] Ainv, np.ndarray[DCTYPE_t, ndim=1] X, int M, int L)

cpdef np.ndarray[DCTYPE_t, ndim=2] iterativeBuildDownRinv(np.ndarray[DCTYPE_t, ndim=2] Ainv, np.ndarray[DCTYPE_t, ndim=1] X, int M, int L)

cpdef np.ndarray[DCTYPE_t, ndim=2] iterativeBuildRinv(np.ndarray[DCTYPE_t, ndim=2] X, int M, int L, int Yavg, double d)