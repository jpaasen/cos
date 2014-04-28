
from libcpp cimport bool

from Complex cimport Complex

ctypedef Complex[float] cuComplex      
      
cdef extern from "BuildR.h":
   cdef cppclass BuildR:
      BuildR()
      
      int getR(
         cuComplex* R,   # buffer holding the resulting covariance matrices
         cuComplex* x,   # buffer holding data vectors      
         float d,        # diagonal loading factor
         int L,          # number of spatial sublengths
         int Yavg,       # number of samples averaged in time
         int M,          # number of data elements 
         int Nx,         # number of data vectors in azimuth
         int Nz,         # number of data vectors in range
         int Nb,         # dimension of beamspace
         # TODO: add suport for custom beamspace matrix
         bool R_on_gpu,  # true if you want R to be left on the gpu
         bool x_on_gpu   # true if x is already on the gpu
         )   
