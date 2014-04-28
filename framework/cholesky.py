'''
Created on 18. aug. 2011

@author: jpaasen
'''

#import math as math
import cmath as cmath
import numpy as np
cimport numpy as np
cimport cython

CTYPE = np.complex
FTYPE = np.float

ctypedef np.complex_t CTYPE_t
ctypedef double complex DCTYPE_t
ctypedef np.float_t FTYPE_t
ctypedef np.int_t ITYPE_t

def solve(A, b, n):
   ''' Solves the positive definite system Ax = b '''
   
   cdef np.ndarray[DCTYPE_t, ndim=2] R = cholesky(A, n)

   cdef np.ndarray[DCTYPE_t, ndim=2] RT = matrixTransposedS(R, n)
   
   cdef np.ndarray[DCTYPE_t, ndim=1] y = forwardSolve(RT, b, n)
   cdef np.ndarray[DCTYPE_t, ndim=1] x = backtrackSolve(R, y, n)
   
   return x

def cholesky(A, n):
   ''' Calculates the Cholesky decomposition of the symmetric matrix A such that R is upper triangular and R'*R = A '''
   
   cdef np.ndarray[DCTYPE_t, ndim=2] R = np.zeros((n,n),dtype=np.complex)
   cdef DCTYPE_t upperColSum
   
   for i in range(n):
      for j in range(n): # TODO: optimization ... switch loops and set i in range(j)
         
         if i == j: # if diagonal element
            
            upperColSum = 0
            for k in range(i):
               upperColSum += R[k,j] * (R[k,j]).conjugate() 
            
            R[i,j] = np.sqrt(A[i,j] - upperColSum)
            
         elif i < j: # if upper triangular element
            
            upperColSum = 0
            for k in range(i):
               upperColSum += R[k,i].conjugate() * R[k,j] 
            
            R[i,j] = (A[i,j] - upperColSum) / R[i,i]
   
   return R


def matrixTransposedS(A, n):
   ''' Transpose the nxn matrix A '''
   C = np.zeros((n,n),dtype=np.complex)
   
   for i in range(n):
      for j in range(n):
         C[j,i] = A[i,j].conjugate()
         
   return C


def forwardSolve(A, b, n):
   ''' Forward solve the lower triangular system Ax = b '''
   cdef np.ndarray[DCTYPE_t, ndim=1] x = np.zeros(n,dtype=np.complex)
   
   for i in range(n):
      
      x[i] = b[i]
      
      for k in range(i):
         x[i] -= A[i,k] * x[k]
      
      x[i] /= A[i,i] 
   
   return x


def backtrackSolve(A, b, n):
   ''' Backtrack solve the upper triangular system Ax = b '''
   cdef np.ndarray[DCTYPE_t, ndim=1] x = np.zeros(n,dtype=np.complex)
   
   for i in reversed(range(n)):
      
      x[i] = b[i]
      
      for k in reversed(range(i+1, n)):
         x[i] -= A[i,k] * x[k]
         
      x[i] /= A[i,i]
   
   return x

  
   
