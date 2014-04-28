
from libcpp cimport bool

from Complex cimport Complex

ctypedef Complex[float] cuComplex      
      
cdef extern from "Capon.h":
   cdef cppclass Capon:
      Capon()
      
      int getCapon(
         cuComplex* z,           # output amplitude per pixel
         cuComplex* w,           # output weights per pixel
         cuComplex* R,           # buffer holding the resulting covariance matrices
         cuComplex* x,           # buffer holding data vectors      
         float d,                # diagonal loading factor
         int L,                  # number of spatial sublengths
         int Yavg,               # number of samples averaged in time
         int M,                  # number of data elements 
         int Nx,                 # number of data vectors in azimuth
         int Nz,                 # number of data vectors in range
         int Nb                  # dimension of beamspace
         )
