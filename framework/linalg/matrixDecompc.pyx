#cython: profile=False
#cython: cdivision=True
#cython: wraparound=False
#cython: boundscheck=False
# filename: matrixDecomp.pyx



import numpy as np # what about mynumpy???
cimport numpy as np

cimport cython

CTYPE = np.complex
FTYPE = np.float
#from framework.cython.types import *


cdef np.ndarray[DCTYPE_t, ndim=2] cholesky(np.ndarray[DCTYPE_t, ndim=2] A, int n):
   ''' 
   Calculates the Cholesky decomposition of the Hermitian matrix A such that U is upper triangular and U'*U = A 
   
   Note: If any U[i,i] is close to zero A is badly conditioned.
   '''
   
   cdef np.ndarray[DCTYPE_t, ndim=2] U
   cdef DCTYPE_t upperColSum
   cdef int i,j,k
   
   U = np.zeros((n,n),dtype=CTYPE)
   
   for i in range(n):
      
      # Handle the diagonal element first
      upperColSum = 0
      for k in range(i):
         upperColSum += U[k,i] * U[k,i].conjugate()
         
      U[i,i] = (A[i,i] - upperColSum)**0.5
      
      # Then handle the upper triangular
      for j in range(i+1,n):
         
         upperColSum = 0
         for k in range(i):
            upperColSum += U[k,i].conjugate() * U[k,j]
            
         U[i,j] = (A[i,j] - upperColSum) / U[i,i]

   return U


cdef uhdu(np.ndarray[DCTYPE_t, ndim=2] A, int n):
   ''' 
   Calculates the UDUH decomposition of the Hermitian matrix A 
   such that U is unit upper triangular, D is diagonal and UDU'=A 
   (' = H = symmetric conjugated)
   
   Now we avoid using the complex sqrt by instead introducing two complex add
   
   Returns [U D]
   
   Note: If any D[i] is close to zero A is badly conditioned.
   '''
   cdef np.ndarray[DCTYPE_t, ndim=2] U
   cdef np.ndarray[DCTYPE_t, ndim=1] D
   cdef DCTYPE_t upperColSum
   cdef int i,j,k
   
   U = np.eye(n, dtype=np.complex)
   D = np.zeros(n, dtype=np.complex)
   
   for i in range(n):
      
      upperColSum = 0
      for k in range(i):
         upperColSum += U[k,i] * U[k,i].conjugate() * D[k]
            
      D[i] = A[i,i] - upperColSum
          
      for j in range(i+1,n): 

         upperColSum = 0
         for k in range(i):
            upperColSum += U[k,i].conjugate() * U[k,j] * D[k]
            
         U[i,j] = (A[i,j] - upperColSum) / D[i]
   
   return U, D#???