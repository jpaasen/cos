
import numpy as np
cimport numpy as np
#cimport cython


ctypedef np.complex_t CTYPE_t
ctypedef double complex DCTYPE_t
ctypedef np.float_t FTYPE_t

#from framework.cython.types cimport *
                                            
cpdef np.ndarray[DCTYPE_t, ndim=1] solveBiCG(np.ndarray[DCTYPE_t, ndim=2] A_in,
                                            np.ndarray[DCTYPE_t, ndim=1] b_in,
                                            np.ndarray[DCTYPE_t, ndim=1] x0_in,
                                            double tol_in,
                                            int itr_in)

cpdef np.ndarray[DCTYPE_t, ndim=1] solveCholesky(np.ndarray[DCTYPE_t, ndim=2] A, np.ndarray[DCTYPE_t, ndim=1] b, int n)


cpdef np.ndarray[DCTYPE_t, ndim=1] solveUHDU(np.ndarray[DCTYPE_t, ndim=2] A, np.ndarray[DCTYPE_t, ndim=1] b, int n)


cpdef np.ndarray[DCTYPE_t, ndim=1] forwardSolve(np.ndarray[DCTYPE_t, ndim=2] A, np.ndarray[DCTYPE_t, ndim=1] b, int n)


cpdef np.ndarray[DCTYPE_t, ndim=1] backtrackSolve(np.ndarray[DCTYPE_t, ndim=2] A, np.ndarray[DCTYPE_t, ndim=1] b, int n)


cpdef np.ndarray[DCTYPE_t, ndim=1] diagonalSolve(np.ndarray[DCTYPE_t, ndim=1] A, np.ndarray[DCTYPE_t, ndim=1] b, int n)