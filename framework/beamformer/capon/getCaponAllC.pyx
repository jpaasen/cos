# filename: caponc.pyx
# line: 0

import framework.mynumpy as np
import scipy as sp

import sys, os

# Select solver to use to compute x = R^-1 a (select only one)
SOLVE_NUMPY    = False # Cholesky based solve in numpy
SOLVE_SCIPY    = False # Cholesky based solve in scipy
SOLVE_CHOLESKY = True  # Our own supreme cholesky solver
SOLVE_UHDU     = False # Our own UHDU decomposition based solver
SOLVE_BICG     = False # Our own Bi-Conjugated Gradient based solver
SOLVE_WOOD     = False # Our own Woodbury matrix inverse identity

BEAMSPACE      = False # Compute beamspace? # TODO: Remove this. Auto detect beamspace. Do beamspace if V is not empty.

DIAGONAL_LOADING_IN_BEAMSPACE = True # If true, diagonal loading is performed after R has been transformed to beamspace.

DEBUG          = False 

###CYTHON_MODE

from framework.beamformer.capon.RealTimeCapon.BuildR cimport BuildR #from gpu.LibRealTimeCapon

from framework.linalg.linearSolvec cimport solveBiCG
from framework.linalg.linearSolvec cimport solveCholesky
from framework.linalg.linearSolvec cimport solveUHDU
from framework.linalg.linearSolvec cimport forwardSolve
from framework.linalg.linearSolvec cimport backtrackSolve
from framework.linalg.linearSolvec cimport diagonalSolve

from framework.linalg.woodburyc cimport iterativeBuildRinv
from framework.linalg.woodburyc cimport iterativeBuildUpRinv
from framework.linalg.woodburyc cimport iterativeBuildDownRinv

cimport numpy as np
cimport cython

#from libcpp cimport bool
from cpython cimport bool

#cdef extern from "/usr/include/google/profiler.h":
#   void ProfilerStart( char* fname )
#   void ProfilerStop()

ctypedef np.complex128_t CTYPE_t
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
cpdef public getCapon(np.ndarray[DCTYPE_t, ndim=2] z_in,       # Output amplitude per pixel
           np.ndarray[DCTYPE_t, ndim=3] w_in,       # Output weights per pixel
           np.ndarray[DCTYPE_t, ndim=2] zPow_in,       # Buffer holding the resulting covariance matrices
           np.ndarray[DCTYPE_t, ndim=3] Xd_in,      # Buffer holding data vectors
           float            regCoef, # Diagonal loading factor
           int              L,       # Size of subarray
           int              Navg,    # 2*Navg+1 is the temporal averaging window
           np.ndarray[DCTYPE_t, ndim=2]  V_in,       # Subspace matrix (assume broken!!!!)
           bool             doFBAvg  # Perform forward-backward averaging? (assume broken!!!!)
           ):
  
   cdef np.ndarray[DCTYPE_t, ndim=2] z
   cdef np.ndarray[DCTYPE_t, ndim=3] w
   cdef np.ndarray[DCTYPE_t, ndim=2] zPow
   cdef np.ndarray[DCTYPE_t, ndim=3] Xd
   cdef np.ndarray[DCTYPE_t, ndim=2] V
   
   cdef int    Nx
   cdef int    Ny
   cdef int    M
      
   #   ProfilerStart("caponc3.prof")
   cdef int    tnp1
   cdef float  K
   cdef float  K_inv
   cdef float  norm
   cdef float  norm_inv
   cdef int    Nb
   cdef int    x,y,i,j,k,g,yy,ii,n
   
   cdef np.ndarray[DCTYPE_t, ndim=1] a
   cdef np.ndarray[DCTYPE_t, ndim=2] ar
   
   cdef np.ndarray[FTYPE_t,  ndim=2] I
   cdef np.ndarray[FTYPE_t,  ndim=2] J
   cdef np.ndarray[DCTYPE_t, ndim=2] R
   cdef np.ndarray[DCTYPE_t, ndim=2] Res
   cdef np.ndarray[DCTYPE_t, ndim=2] Rbs
   cdef np.ndarray[DCTYPE_t, ndim=2] Rinv
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
    
   cdef bool       useSubspace
    
   # Assign outputs
   z    = z_in
   zPow = zPow_in
   w    = w_in
  
   ###CYTHON_MODE_END§
   """PYTHON_MODE
def getCaponCPy( Xd_in,      # Buffer holding data vectors
              regCoef, # Diagonal loading factor
              L,       # Size of subarray
              Navg,    # 2*Navg+1 is the temporal averaging window
              V_in,       # Subspace matrix (assume broken!!!!)
              doFBAvg,  # Perform forward-backward averaging? (assume broken!!!!)
              verbose
            ):
   
   z    = np.zeros((Xd_in.shape[0], Xd_in.shape[1]-2*Navg),    dtype=np.complex128)
   
   if BEAMSPACE:
      w    = np.zeros((Xd_in.shape[0], Xd_in.shape[1]-2*Navg, V_in.shape[0]), dtype=np.complex128)
   else:
      w    = np.zeros((Xd_in.shape[0], Xd_in.shape[1]-2*Navg, L), dtype=np.complex128)
   
   zPow = np.zeros((Xd_in.shape[0], Xd_in.shape[1]-2*Navg),    dtype=np.complex128)
   

PYTHON_MODE_END"""

   Xd   = Xd_in
   V    = V_in

   Nx                = Xd_in.shape[0]
   Ny                = Xd_in.shape[1]
   M                 = Xd_in.shape[2]
   tnp1              = 2*Navg+1
   K                 = M-L+1
   K_inv             = 1.0/K
   norm              = K * tnp1
   norm_inv          = 1.0 / norm
   Nb                = V.shape[0]
   y = 0
   i = 0
   
  
   ar               = np.zeros((2*Navg+1, M), dtype=np.complex128)
      
   I                = np.eye(L)
   J                = np.rot90(I)
   

   Res              = np.zeros((L,L), dtype=np.complex128)
   g_singleSnapshotes = np.zeros((L,), dtype=np.complex128)
   
   
   if BEAMSPACE:
      Rbs           = np.zeros((Nb,Nb), dtype=np.complex128)
      R             = np.zeros((Nb,Nb), dtype=np.complex128)
      g_singleSnapshot = np.zeros((Nb,), dtype=np.complex128)
      a             = np.ones((Nb,), dtype=np.complex128)
   else:
      R             = np.zeros((L,L), dtype=np.complex128)
      Ria           = np.zeros((L,), dtype=np.complex128)
      g_singleSnapshot = np.zeros((L,), dtype=np.complex128)
      a             = np.ones((L,), dtype=np.complex128)

      
   Rinv             = np.zeros((L,L), dtype=np.complex128)
   R_               = np.zeros((L,L), dtype=np.complex128)
   R_empty          = np.zeros((L,L), dtype=np.complex128)
   subarr_sums      = np.zeros((L,), dtype=np.complex128)
   R_full           = np.zeros((M,M), dtype=np.complex128)

   dot_sum = 0
   percent = 0

   R_n              = np.zeros((L,L,2*Navg+2), dtype=np.complex128)
      
   useSubspace = BEAMSPACE
   #useSubspace = (V.ndim == 2)
   if BEAMSPACE:
      if DEBUG:
         print 'Capon algorithm "subspaced" down to %d dims.' %Nb
      
      # The column of ones (what we seek the 'magnitude' of) represented in the subspace
      a = np.dot(V, np.ones((L,), dtype=np.complex128))
      
       
  
   # Iterate over all x-pixels
   for x in range(Nx):
           
      # Iterate over all y-pixels, except the boundaries where full time averaging is impossible 
      for y in range(Navg, Ny-Navg):
         
         if SOLVE_WOOD: # with woodbury we are building the inverse of R instead of R
         
            ###CYTHON_MODE
            if y == 0: # make Rinv using the whole time avg area
               for i in range(2*Navg+1):
                  for j in range(M):
                     ar[i,j] = Xd[x,i+y-Navg,j]
               Rinv = iterativeBuildRinv(ar, M, L, Navg, regCoef)
               
            else: # update Rinv using sliding window in fast time
               Rinv = iterativeBuildUpRinv(Rinv, Xd[x,y-Navg-1,:], M, L)
               Rinv = iterativeBuildDownRinv(Rinv, Xd[x,y+Navg,:], M, L)
            ###CYTHON_MODE_END§
            """PYTHON_MODE
            pass
            PYTHON_MODE_END"""
            
         
         else: # if SOLVE_WOOD is not selected we build R
           
            ## Select data for pixel (y,x); i.e. 2n+1 time values around 'y' for all sensor channels
            #for i in range(2*Navg+1):
               #for j in range(M):
                  #ar[i,j] = Xd[x,i+y-Navg,j]
                
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
                  Res[i,j] = 0
            # Iterate over subarrays
            for i in range(M-L+1):
               # Iterate over rows in R
               for j in range(L):
                  # Iterate over columns in R
                  for k in range(j,L):
                     # Sum the products
                     Res[j,k] = Res[j,k] + R_full[i+j,i+k]
   
            # Normalise R = R.conj() / norm
            for j in range(L):
               for k in range(j,L):
                  Res[j,k] = Res[j,k].conjugate() * norm_inv
#                  
#            if BEAMSPACE:
#               print "More"
#               print Navg
#               print ar
#               print Xd[x,y-Navg:y+Navg+1,:]
#               print V
#               print "V.shape"
#               #print V.shape
#               Rbs = np.dot(np.dot(V,Xd[x,y-Navg:y+Navg+1,:]),V.T.conj())
   
         
         # Because of symmetry, only half of R was computed. Restore the last half:
         # This is only needed when we're using the official solve-functions, our
         # Cholesky-based solve does not require the lower triangular of R
         # Iterate over rows in R
         """PYTHON_MODE
         SOLVE_NUMPY = True
         PYTHON_MODE_END"""
         if SOLVE_NUMPY:
            for i in range(L):
               # Iterate over columns in R
               for j in range(i+1,L):
                  # Sum the products
                  Res[j,i] = Res[i,j].conjugate()
   
            
         # R contains an estimate of the covariance matrix
         # Store the sum of the current-time outputs in 'g_singleSnapshot':
         subarr_sums[0] = 0
         for i in range(M-L+1):
            subarr_sums[0] = subarr_sums[0] + Xd[x,y,i]
         for i in range(1, L):
            ii = i-1
            subarr_sums[i] = subarr_sums[ii] - Xd[x,y,ii] + Xd[x,y,M-L+i]
   
         for i in range(L):
            g_singleSnapshotes[i] = subarr_sums[i] * K_inv
   
         
         if doFBAvg:
            Res = 0.5 * ( Res + np.dot(np.dot(J, Res.T), J) )
         
        
         if not DIAGONAL_LOADING_IN_BEAMSPACE:        
            # We choose to apply diagonal loading to the elementspace R, regardless of whether
            # beamspace is implemented or not. This implies that a fake noise signal is added
            # evenly to all beams, and that the beamspace method will shave some of this noise
            # off (along with the signal/noise energy that was already present there).
            if SOLVE_WOOD:
               pass
            else:        
               R_trace = 0
               for i in range(L): R_trace = R_trace + Res[i,i]
               diag_load = R_trace * regCoef / L
               for i in range(L):
                  Res[i,i] = Res[i,i] + diag_load
            
         
         # If a subspace matrix V is given and we _are_ using FB averaging, we
         # have to wait until now to go to the reduced space:
         if BEAMSPACE:
            R = np.dot(np.dot(V, Res), V.T.conj())
            g_singleSnapshot = np.dot(V, g_singleSnapshotes)
            LL = Nb
         else:
            R = Res
            g_singleSnapshot = g_singleSnapshotes
            LL = L
   
         if DIAGONAL_LOADING_IN_BEAMSPACE:
            # Apply diagonal loading if a solver other than WOOD is selected  
            if SOLVE_WOOD:
               pass
            else:        
               R_trace = 0
               for i in range(LL): R_trace = R_trace + R[i,i]
               diag_load = R_trace * regCoef / LL
               for i in range(LL):
                  R[i,i] = R[i,i] + diag_load
            
         ####################
         # SOLVE x = R^-1 a #
         # Since R is both symmetric and positive-definite we don't need a general solver:
         
         ###CYTHON_MODE           

         # Our own supreme cholesky solver
         if SOLVE_CHOLESKY:
            Ria = solveCholesky(R, a, LL)
            
         # With Woodbury we already have the inverse and can just multiply 
         elif SOLVE_WOOD:
            for i in range(LL):
               for j in range(LL):
                  Ria[i] = Ria[i] + Rinv[i,j] * a[j]
                  
         # LU based solve in numpy
         elif SOLVE_NUMPY:
            Ria = np.linalg.solve(R, a)
   
         # Cholesky based solve in scipy
         elif SOLVE_SCIPY:
            #Ria = cho_solve(cho_factor(R), a)
            print 'Scipy solver not suported'
            pass
                    
         # Our own UHDU decomposition based solver
         elif SOLVE_UHDU:
            Ria = solveUHDU(R, a, LL)
            
         # Our own Bi-Conjugated Gradient based solver
         elif SOLVE_BICG:
            #Ria = solveBiCG(R, a, Ria, 0.0, L/6) # Search for solution startes from Ria. Only L/6 iterations are performed.
            Ria = solveBiCG(R, a, np.zeros(LL, dtype=complex), 0.0, 0)
            
         else:
            print 'No solve function specified! Aborting.'
            Exception('Can not continue from here...')
            
         ###CYTHON_MODE_END§
         """PYTHON_MODE
         Ria = np.linalg.solve(R, a)
         PYTHON_MODE_END"""
         
         # END #
         
          
         aRia = 0
         for i in range(LL):
            aRia = aRia + a[i].conjugate()*Ria[i]
         for i in range(LL):
            w[x,y-Navg,i] = Ria[i] / aRia
         
         for i in range(LL):
            z[x,y-Navg] = z[x,y-Navg] + w[x,y-Navg,i].conjugate()*g_singleSnapshot[i] # Note: A bit ad-hoc maybe, but uses only the current time-snapshot to calculate the output/'alpha' value
            
         for i in range(LL):
            idot = 0
            for j in range(LL):
               idot = idot + w[x,y-Navg,j].conjugate()*R[i,j]
            zPow[x,y-Navg] = zPow[x,y-Navg] + idot*w[x,y-Navg,i]

           
#################
## Python mode ##
#################
 
   return [z, zPow, w]
       
#   ProfilerStop()