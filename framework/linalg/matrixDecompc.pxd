#cython: profile=False
#cython: cdivision=True
#cython: wraparound=False
#cython: boundscheck=False
# filename: matrixDecomp.pyx


import numpy as np # what about mynumpy???
cimport numpy as np

cimport cython

CTYPE = np.complex
FTYPE = np.float

ctypedef np.complex_t CTYPE_t
ctypedef double complex DCTYPE_t
ctypedef np.float_t FTYPE_t
#from framework.cython.types cimport *

cdef np.ndarray[DCTYPE_t, ndim=2] cholesky(np.ndarray[DCTYPE_t, ndim=2] A, int n)

cdef uhdu(np.ndarray[DCTYPE_t, ndim=2] A, int n)