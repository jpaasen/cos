# filename: caponc.pyx
# line: 0

import numpy as np
import scipy as sp
cimport numpy as np

np.set_printoptions(precision=8)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=300)
np.set_printoptions(threshold=10000)

import sys, os

from framework.linalg.linearSolvec cimport solveBiCG
from framework.linalg.linearSolvec cimport solveCholesky
from framework.linalg.linearSolvec cimport solveUHDU
from framework.linalg.linearSolvec cimport forwardSolve
from framework.linalg.linearSolvec cimport backtrackSolve
from framework.linalg.linearSolvec cimport diagonalSolve

from framework.linalg.woodburyc cimport iterativeBuildRinv
from framework.linalg.woodburyc cimport iterativeBuildUpRinv
from framework.linalg.woodburyc cimport iterativeBuildDownRinv
#os.chdir(current_path)

from Complex cimport *

# Select solver to use to compute x = R^-1 a (select only one)
# TODO: Change how this is done!
SOLVE_NUMPY    = False # Cholesky based solve in numpy
SOLVE_SCIPY    = False # Cholesky based solve in scipy
SOLVE_CHOLESKY = True  # Our own supreme cholesky solver
SOLVE_UHDU     = False # Our own UHDU decomposition based solver
SOLVE_BICG     = False # Our own Bi-Conjugated Gradient based solver
SOLVE_WOOD     = False # Our own Woodbury matrix inverse identity

USE_MKL        = False

DEBUG          = False

cimport cython
cimport cython.view
cimport cython.array
ctypedef Complex[double] ComplexDouble    

#from libcpp cimport bool
#from cpython cimport bool
from libcpp cimport bool

#cdef extern from "/usr/include/google/profiler.h":
#   void ProfilerStart( char* fname )
#   void ProfilerStop()

ctypedef np.complex_t CTYPE_t
ctypedef double complex DCTYPE_t
ctypedef np.float_t FTYPE_t
ctypedef np.int_t ITYPE_t

# Python entry point
cpdef getCaponCPy(np.ndarray Xd_in,
               np.float   regCoef_in,
               np.int     L_in,
               np.int     Navg_in,
               np.ndarray V_in,
               bool       doFBAvg,
               bool       verbose):
   
   if DEBUG:
      print "\033[01;34mXd in getCaponPy\033[00m"
      print Xd_in
      
   #print "Entry Python Interface"
   cdef int    Nx      = Xd_in.shape[0]
   cdef int    Ny      = Xd_in.shape[1]
   cdef int    M       = Xd_in.shape[2]
   cdef int    Navg    = Navg_in
   cdef int    L       = L_in
   cdef float  regCoef = regCoef_in

   cdef np.ndarray[DCTYPE_t, ndim=2] z_py    = np.zeros((Nx, Ny-2*Navg),           dtype=np.complex128)
   cdef np.ndarray[DCTYPE_t, ndim=3] w_py    = np.zeros((Nx, Ny-2*Navg, L), dtype=np.complex128)
   cdef np.ndarray[DCTYPE_t, ndim=2] zPow_py = np.zeros((Nx, Ny-2*Navg),           dtype=np.complex128)
   cdef np.ndarray[DCTYPE_t, ndim=3] Xd_py   = Xd_in
   
   cdef np.ndarray[DCTYPE_t, ndim=2] V_py
   if V_in.ndim < 2:
      V_py = np.array([[]], dtype=np.complex128)
   else:
      V_py = V_in   

   getCapon(z_py, w_py, zPow_py, Xd_py, regCoef, L, Navg, V_py, doFBAvg)

   return [z_py, zPow_py, w_py]

   
# C interface
cpdef public getCapon(np.ndarray[DCTYPE_t, ndim=2] z_in,    # Output amplitude per pixel
                      np.ndarray[DCTYPE_t, ndim=3] w_in,    # Output weights per pixel
                      np.ndarray[DCTYPE_t, ndim=2] zPow_in, # Buffer holding the resulting covariance matrices
                      np.ndarray[DCTYPE_t, ndim=3] Xd_in,   # Buffer holding data vectors
                      float                        regCoef, # Diagonal loading factor
                      int                          L,       # Size of subarray
                      int                          Navg,    # 2*Navg+1 is the temporal averaging window
                      np.ndarray[DCTYPE_t, ndim=2] V_in,    # Subspace matrix
                      bool                         doFBAvg  # Perform forward-backward averaging? (assume broken!!!!)
):

   #print "Entry C Interface"
      
   cdef np.ndarray[DCTYPE_t, ndim=2] z    = z_in
   cdef np.ndarray[DCTYPE_t, ndim=3] w    = w_in
   cdef np.ndarray[DCTYPE_t, ndim=2] zPow = zPow_in
   cdef np.ndarray[DCTYPE_t, ndim=3] Xd   = Xd_in
   cdef np.ndarray[DCTYPE_t, ndim=2] V    = V_in
   
   if DEBUG:
      print "\033[01;34mXd in getCaponC.pyx\033[00m"
      print Xd_in
   
   cdef int    Nx      = Xd_in.shape[0]
   cdef int    Ny      = Xd_in.shape[1]
   cdef int    M       = Xd_in.shape[2]
   
   if DEBUG:
      print "M:%d  d:%2.2f  L:%d  Navg=%d  Nx:%d  Ny:%d"%(M,regCoef,L,Navg,Nx,Ny)

   cdef int    tnp1
   cdef float  K
   cdef float  K_inv
   cdef float  norm
   cdef float  norm_inv
   cdef int    nSubspaceDims
   cdef int    x,y,i,j,k,g,yy,ii,n
   
   cdef np.ndarray[DCTYPE_t, ndim=1] a
   cdef np.ndarray[DCTYPE_t, ndim=2] ar

   cdef np.ndarray[FTYPE_t,  ndim=2] I
   cdef np.ndarray[FTYPE_t,  ndim=2] J
   cdef np.ndarray[DCTYPE_t, ndim=2] Rinv
   cdef np.ndarray[DCTYPE_t, ndim=2] R
   cdef np.ndarray[DCTYPE_t, ndim=2] R_
   cdef np.ndarray[DCTYPE_t, ndim=2] R_empty
   cdef np.ndarray[DCTYPE_t, ndim=1] subarr_sums
   cdef np.ndarray[DCTYPE_t, ndim=2] R_full
   cdef np.ndarray[DCTYPE_t, ndim=1] g_singleSnapshot
   cdef np.ndarray[DCTYPE_t, ndim=1] Ria

   cdef DCTYPE_t dot_sum
   cdef DCTYPE_t ar_conj, R_trace, diag_load, aRia, idot, csum
   cdef FTYPE_t percent

   cdef np.ndarray[DCTYPE_t, ndim=3] R_n
    
   if DEBUG:
      print "Assigning"
#   cdef bool       useSubspace


   if USE_MKL:
      from framework.lib.mkl.mklcSolveCholeskyC import mklcSolveCholeskyC
   
   tnp1              = 2*Navg+1
   K                 = M-L+1
   K_inv             = 1.0/K
   norm              = K * tnp1
   norm_inv          = 1.0 / norm
   nSubspaceDims     = V.shape[0]
   y = 0
   i = 0
   
   if DEBUG:
      print "Allocating memory"
   
   a                = np.ones((L,), dtype=np.complex128)
   ar               = np.zeros((2*Navg+1, M), dtype=np.complex128)
   I                = np.eye(L)
   J                = np.rot90(I)
   R                = np.zeros((L,L), dtype=np.complex128)
   Rinv             = np.zeros((L,L), dtype=np.complex128)
   R_               = np.zeros((L,L), dtype=np.complex128)
   R_empty          = np.zeros((L,L), dtype=np.complex128)
   subarr_sums      = np.zeros((L,), dtype=np.complex128)
   R_full           = np.zeros((M,M), dtype=np.complex128)
   g_singleSnapshot = np.zeros((L,), dtype=np.complex128)
   Ria              = np.zeros((L,), dtype=np.complex128)
   dot_sum = 0
   percent = 0

   R_n              = np.zeros((L,L,2*Navg+2), dtype=np.complex128)
   
   if DEBUG:
      print "Start calculations of Capon using cython ..."
    
   # Iterate over all x-pixels
   for x in range(Nx):     
           
      # Iterate over all y-pixels, except the boundaries where full time averaging is impossible 
      for y in range(Navg, Ny-Navg):
         
         # Select data for pixel (y,x); i.e. 2n+1 time values around 'y' for all sensor channels
         for i in range(2*Navg+1):
            for j in range(M):
               ar[i,j] = Xd[x,i+y-Navg,j]
               
         # Compute R_full = dot(ar.T.conj(), ar), but take symmetry into account:
         for i in range(M-L+1):
            for j in range(L):
               dot_sum = 0
               for k in range(2*Navg+1):
                  dot_sum = dot_sum + ar[k,i].conjugate()*ar[k,i+j]
               R_full[i,i+j] = dot_sum
         
         # The above let a lower right triangular part of R_full empty. Handle that now..
         g = 1
         for i in range(M-L+1,M):
            for j in range(L-g):
               dot_sum = 0
               for k in range(2*Navg+1):
                  dot_sum = dot_sum + ar[k,i].conjugate()*ar[k,i+j]
               R_full[i,i+j] = dot_sum
            g = g + 1

               
         # Sum the subarray covariance matrices
         for i in range(L):
            for j in range(L):
               R[i,j] = 0
         # Iterate over subarrays
         for i in range(M-L+1):
            # Iterate over rows in R
            for j in range(L):
               # Iterate over columns in R
               for k in range(j,L):
                  # Sum the products
                  R[j,k] = R[j,k] + R_full[i+j,i+k]

         # Normalise R = R.conj() / norm
         for j in range(L):
            for k in range(j,L):
               R[j,k] = R[j,k].conjugate() * norm_inv

                    
         # R contains an estimate of the covariance matrix
         # Store the sum of the current-time outputs in 'g_singleSnapshot':
         subarr_sums[0] = 0
         for i in range(M-L+1):
            subarr_sums[0] = subarr_sums[0] + Xd[x,y,i]
         for i in range(1, L):
            ii = i-1
            subarr_sums[i] = subarr_sums[ii] - Xd[x,y,ii] + Xd[x,y,M-L+i]

         for i in range(L):
            g_singleSnapshot[i] = subarr_sums[i] * K_inv

         # Apply diagonal loading
         R_trace = 0
         for i in range(L): R_trace = R_trace + R[i,i]
         diag_load = R_trace * regCoef / L
         for i in range(L):
            R[i,i] = R[i,i] + diag_load

            
         ####################
         # SOLVE x = R^-1 a #
         # Since R is both symmetric and positive-definite we don't need a general solver:


         # Our own supreme cholesky solver
         if USE_MKL:
            Ria = mklcSolveCholeskyC(R, a)
         else:
            Ria = solveCholesky(R, a, L)

         aRia = 0
         for i in range(L):
            aRia = aRia + a[i].conjugate()*Ria[i]
         for i in range(L):
            w[x,y-Navg,i] = Ria[i] / aRia
         
         for i in range(L):
            z[x,y-Navg] = z[x,y-Navg] + w[x,y-Navg,i].conjugate()*g_singleSnapshot[i] # Note: A bit ad-hoc maybe, but uses only the current time-snapshot to calculate the output/'alpha' value
            
         for i in range(L):
            idot = 0
            for j in range(L):
               idot = idot + w[x,y-Navg,j].conjugate()*R[i,j]
            zPow[x,y-Navg] = zPow[x,y-Navg] + idot*w[x,y-Navg,i]
           
      if False:
         percent = np.round( 100* (y-Navg) / (Ny-2*Navg))
         if np.mod(y, 5) == 0:
            print ' %d\%'%percent
            
         if np.mod(y, 100) == 0:
            pass
      
   if DEBUG:
      print "z in getCaponC.pyx"
      print z
    
#   ProfilerStop()
#   return [z, zPow, w]