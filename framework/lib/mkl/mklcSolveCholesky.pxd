from libcpp cimport bool
cimport numpy as np

#ctypedef np.float64_t FTYPE_t

#cdef extern from "mkl.h":
#ctypedef double complex MKL_Complex16
from Complex cimport *
#ctypedef Complex[double] CTYPE

cdef extern from "mklcSolveCholesky.h":
   int mlkcSolveCholesky(Complex[double] *a_py, Complex[double] *b, int order)

#template <class T>
#int mklUnivariateSpline(int Nx,  T *x, T *y,
#                        int Nyi, T *xi, T *yi,
#                        int kind, bool xi_uniform )