
import numpy as np
import scipy as sp
cimport numpy as np

from libcpp cimport bool

cimport mklcSolveCholesky as solve
from Complex cimport *
ctypedef Complex[double] DoubleComplex

FTYPE_s = 'float64'
FTYPE = np.float64
ctypedef np.float64_t FTYPE_t

CTYPE_s = 'complex128'
CTYPE = np.complex128
ctypedef np.complex128_t CTYPE_t

def mklcSolveCholeskyC(np.ndarray A_in, np.ndarray b_in):

   cdef int order = A_in.shape[0]
   
   cdef np.ndarray[CTYPE_t, ndim=2] A_py = A_in#.astype(CTYPE_s)
   cdef np.ndarray[CTYPE_t, ndim=1] b_py = b_in#.astype(CTYPE_s)
   
#   cdef np.ndarray[FTYPE_t, ndim=1] c_py = np.zeros(order, dtype=CTYPE)
   
   cdef DoubleComplex *A = <DoubleComplex*>A_py.data
   cdef DoubleComplex *b = <DoubleComplex*>b_py.data
#   cdef CTYPE_t *c = <CTYPE_t*>c_py.data
   
   solve.mlkcSolveCholesky(A,b,order)
   
#   print b_py

   return b_py
#                
