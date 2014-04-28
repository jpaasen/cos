#cython: profile=False
#cython: cdivision=True
#cython: wraparound=False
#cython: boundscheck=False

import numpy as np
cimport numpy as np
cimport cython

from cpython cimport bool 

CTYPE = np.complex
FTYPE = np.float

cpdef np.ndarray[DCTYPE_t, ndim=2] iterativeBuildUpRinv(np.ndarray[DCTYPE_t, ndim=2] Ainv, np.ndarray[DCTYPE_t, ndim=1] X, int M, int L):
   '''
   Update Ainv with the data in X, using sub-arrays of length L.
   Remember to normalize X before calling this function.
   '''
   newAinv = np.array(Ainv)
   
   K = M - L + 1;
   
   for l in range(K):
      newAinv = shermanMorrisonUp(X[l:l+L], newAinv, L)
         
   return newAinv

cpdef np.ndarray[DCTYPE_t, ndim=2] iterativeBuildDownRinv(np.ndarray[DCTYPE_t, ndim=2] Ainv, np.ndarray[DCTYPE_t, ndim=1] X, int M, int L):
   '''
   Down grade Ainv with the data in X, using sub-arrays of length L.
   Remember to normalize X before calling this function.
   '''
 
   newAinv = np.array(Ainv)
 
   K = M - L + 1;
   
   for l in range(K):
      newAinv = shermanMorrisonDown(X[l:l+L], newAinv, L)
      
   return newAinv
         

cpdef np.ndarray[DCTYPE_t, ndim=2] iterativeBuildRinv(np.ndarray[DCTYPE_t, ndim=2] X, int M, int L, int Yavg, double d):
   '''
   Builds Rinv from the data area X
   using M elements, L long sub arrays, 2*Yavg+1 time samples and diagonal loading d.
   X needs to be of size (2*Yavg+1, M).
   
   The inverse is normalized with 1 / ((2*Yavg+1) * (M-l+1))
   '''
   diagonalLoadingFactor = 1
   if d > 0:  # if d is 0, use 1. Should use a smaller value, but it is not possible due to poor conditioning
      diagonalLoadingFactor = 1 / d
   
   Rinv = (diagonalLoadingFactor * np.eye(L));
   
   N = 2*Yavg + 1;
   K = M - L + 1;
   
   for n in range(N):
      Rinv = iterativeBuildUpRinv(Rinv, X[n], M, L)
      
   Rinv = (N * K) * Rinv # normalize Rinv
   
   return Rinv; 


cdef inline np.ndarray[DCTYPE_t, ndim=2] shermanMorrison(np.ndarray[DCTYPE_t, ndim=1] x, np.ndarray[DCTYPE_t, ndim=2] Ainv, int m, bool up):
   ''' 
   Does a rank-1 adjustment of Ainv using the Sherman-Morrison formula
   (A + xx^H)^(-1) = A^(-1) - (A^(-1)xx^(H)A^(-1)) / (1 + x^(H)A^(-1)x)
   if (1 + x^(H)A^(-1)x) != 0.
   
   Ainv is a matrix of size m*m
   x is a column vector of length m    
   
   If up == True: A rank update is performed, 
   otherwise we do a rank degrade.
   ''' 
   factor = 1
   if up != True: 
      factor = -1
   
   xH = x.conjugate()
   Ainvx = np.dot(Ainv, x)
   xHAinv = np.dot(xH, Ainv)
   xHAinvx = np.dot(xH, np.dot(Ainv, x)) # memory layout impose this product rather than reusing xHAinv (the result here is a scalar)
   
   numerator = np.outer(Ainvx, xHAinv)
   denominator = 1 + factor * xHAinvx
   if abs(denominator) == 0:
      raise Exception('Denominator is 0 in Sherman Morrison formula')
   
   return Ainv - ( (factor/denominator) * numerator )


cdef inline np.ndarray[DCTYPE_t, ndim=2] shermanMorrisonUp(np.ndarray[DCTYPE_t, ndim=1] x, np.ndarray[DCTYPE_t, ndim=2] Ainv, int m):
   ''' Does a rank-1 upgrade of Ainv using the Sherman-Morrison formula '''
   cdef np.ndarray[DCTYPE_t, ndim=2] newAinv = shermanMorrison(x, Ainv, m, True)
   return newAinv

cdef inline np.ndarray[DCTYPE_t, ndim=2] shermanMorrisonDown(np.ndarray[DCTYPE_t, ndim=1] x, np.ndarray[DCTYPE_t, ndim=2] Ainv, int m):
   ''' Does a rank-1 downgrade of Ainv using the Sherman-Morrison formula '''
   cdef np.ndarray[DCTYPE_t, ndim=2] newAinv = np.zeros((m,m), dtype=np.complex)
   newAinv = shermanMorrison(x, Ainv, m, False)
   return newAinv