
import framework.mynumpy as np
cimport numpy as np

# TODO: Rename Capon into CaponCUDA
from Capon cimport *
cdef Capon capon

# Python entry point
cpdef getCaponCUDAPy(np.ndarray Xd_in,
                     np.float   d_in,
                     np.int     L_in,
                     np.int     Navg_in,
                     np.ndarray V_in,
                     bool       doFBAvg,
                     bool       verbose):
   
    
   #print "Entry Python Interface"
   cdef int    Nx      = Xd_in.shape[0]
   cdef int    Ny      = Xd_in.shape[1]
   cdef int    M       = Xd_in.shape[2]
   cdef int    Navg    = Navg_in
   cdef int    L       = L_in
   cdef int    Nb      = 0
   cdef float  d       = d_in

   cdef np.ndarray[np.complex64_t, ndim=2] z_py    = np.zeros((Nx, Ny-2*Navg),       dtype=np.complex64)
   cdef np.ndarray[np.complex64_t, ndim=3] w_py    = np.zeros((Nx, Ny-2*Navg, L),    dtype=np.complex64)
   cdef np.ndarray[np.complex64_t, ndim=4] R_py    = np.zeros((Nx, Ny-2*Navg, L, L), dtype=np.complex64)
   cdef np.ndarray[np.complex64_t, ndim=3] Xd_py   = Xd_in.astype('complex64')
   
   if V_in.ndim == 2:
      Nb   = V_in.shape[0]
    
   #print "Beamspace"
   #print Nb

   z  = <Complex[float]*>z_py.data
   w  = <Complex[float]*>w_py.data
   R  = <Complex[float]*>R_py.data
   Xd = <Complex[float]*>Xd_py.data

#   # Allocate memory for the capon solver on the heap:
#   # (used to be "cdef Capon capon" for the stack, but this lead object cleanup issues)
#   cdef Capon *capon = new Capon()
#   try:
#      e = capon.getCapon(
#         z,                # output amplitude per pixel
#         w,                # output weights per pixel
#         R,                # buffer holding the resulting covariance matrices
#         Xd,               # buffer holding data vectors      
#         d,                # diagonal loading factor
#         L,                # number of spatial sublengths
#         Navg,             # number of samples averaged in time
#         M,                # number of data elements 
#         Nx,               # number of data vectors in azimuth
#         Ny,               # number of data vectors in range
#         Nb                # dimension of beamspace
#         )
#   finally:
#      # Do not leave object cleanup to Python, by ensure it here:
#      del capon

   e = capon.getCapon(
      z,                # output amplitude per pixel
      w,                # output weights per pixel
      R,                # buffer holding the resulting covariance matrices
      Xd,               # buffer holding data vectors      
      d,                # diagonal loading factor
      L,                # number of spatial sublengths
      Navg,             # number of samples averaged in time
      M,                # number of data elements 
      Nx,               # number of data vectors in azimuth
      Ny,               # number of data vectors in range
      Nb                # dimension of beamspace
      )
      
#   Capon.getCapon(z_py, w_py, R_py, Xd_py, regCoef, L, Navg, M, Nx, Ny, Nb)

   return [z_py, R_py, w_py, e]
