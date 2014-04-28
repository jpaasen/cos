
import numpy as np
cimport numpy as np
from libcpp cimport bool

from Complex cimport Complex
      
#ctypedef Complex[float] cuComplex
      
      
cdef extern from "Solver.h":
   cdef cppclass Solver:
      Solver()
      
      int solve(
         cuComplex* x,           # buffer holding the solutions
         cuComplex* A,     # buffer holding matrices
         cuComplex* b,     # buffer holding the left sides      
         int N,            # size of each linear system
         int batch,        # number of linear systems
         bool x_on_gpu,    # true if x should remain on the gpu
         bool A_on_gpu,    # true if R is already on the gpu
         bool b_on_gpu     # true if b is already on the gpu
         )


def solveGaussJordanGPU( A_in, b_in ):
      
   cdef int  L        = A_in.shape[0]
   cdef int  N        = L        # size of each linear system
   cdef int  batch    = 1        # number of linear systems
   cdef bool x_on_gpu = False    # true if x should remain on the gpu
   cdef bool A_on_gpu = False    # true if R is already on the gpu
   cdef bool b_on_gpu = False    # true if b is already on the gpu
   
   cdef np.ndarray[np.complex64_t, ndim=2] A_py  = A_in.astype('complex64')
   cdef cuComplex* A = <cuComplex*>A_py.data
   
   cdef np.ndarray[np.complex64_t, ndim=1] b_py  = b_in.astype('complex64')
   cdef cuComplex* b = <cuComplex*>b_py.data
   
   cdef np.ndarray[np.complex64_t, ndim=2] x_py = np.zeros((batch,L),dtype=np.complex64)
   cdef cuComplex* x = <cuComplex*>x_py.data
   
   cdef Solver solver  

   solver.solve(
      x,             # buffer holding the solutions
      A,             # buffer holding matrices
      b,             # buffer holding the left sides      
      N,             # size of each linear system
      batch,         # number of linear systems
      x_on_gpu,      # true if x should remain on the gpu
      A_on_gpu,      # true if R is already on the gpu
      b_on_gpu       # true if b is already on the gpu
      )
   
   return x_py