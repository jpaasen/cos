
from libcpp cimport bool

from Complex cimport Complex
      
ctypedef Complex[float] cuComplex
      
      
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
