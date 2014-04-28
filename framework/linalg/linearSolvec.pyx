#cython: profile=False
#cython: cdivision=True
#cython: wraparound=False
#cython: boundscheck=False
# filename: linearSolve.pyx


import sys, os

#PHDCODE_ROOT = os.environ['PHDCODE_ROOT']
#sys.path.append("%s/python"%PHDCODE_ROOT)

# TODO: import matrixDecomp

from matrixDecompc cimport cholesky
from matrixDecompc cimport uhdu

import numpy as np
#from framework.linalg.woodbury import den
cimport numpy as np
cimport cython

CTYPE = np.complex
FTYPE = np.float

#from framework.cython.types cimport *
#ctypedef np.complex_t CTYPE_t
#ctypedef double complex DCTYPE_t
#ctypedef np.float_t FTYPE_t

#cdef np.ndarray[DCTYPE_t, ndim=1] solveBiCG(np.ndarray[DCTYPE_t, ndim=2] A,
#                                            np.ndarray[DCTYPE_t, ndim=1] b,
#                                            np.ndarray[DCTYPE_t, ndim=1] x0 = 0,
#                                            double tol = 0,
#                                            int itr = 0):
#                                            
#cdef np.ndarray[DCTYPE_t, ndim=1] solveBiCG(A,
#                                            b,
#                                            x0 = 0,
#                                            tol = 0,
#                                            itr = 0):
                       
#cdef solveBiCG(np.ndarray A,
#               np.ndarray b,
#               np.ndarray x0,
#               double tol,
#               int itr):
   
cdef inline np.ndarray[DCTYPE_t, ndim=2] matrixTransposedS(np.ndarray[DCTYPE_t, ndim=2] A, int n):
   ''' Transpose the nxn matrix A '''
   cdef np.ndarray[DCTYPE_t, ndim=2] C = np.zeros((n,n),dtype=np.complex)
   cdef int i,j
   
   for i in range(n):
      for j in range(n):
         C[j,i] = A[i,j].conjugate()
         
   return C
#
                                            
cpdef np.ndarray[DCTYPE_t, ndim=1] solveBiCG(np.ndarray[DCTYPE_t, ndim=2] A_in,
                                            np.ndarray[DCTYPE_t, ndim=1] b_in,
                                            np.ndarray[DCTYPE_t, ndim=1] x0_in,
                                            double tol_in,
                                            int itr_in):
   
   ''' Solves the complex linear system Ax = b using the complex-biconjugate-gradient method
      
      Arguments:
      A   -- Coefficient matrix
      b   -- Right-side vector
      x0  -- Initial solution (default: 0-vector)
      tol -- Stop solver when error is less than tol (default: 0)
      itr -- Stop solver after itr iterations -- default
   '''
   cdef np.ndarray[DCTYPE_t, ndim=2] A
   cdef np.ndarray[DCTYPE_t, ndim=1] b
   cdef np.ndarray[DCTYPE_t, ndim=1] x0
   cdef double tol
   cdef int itr
   
   cdef int m
   cdef int n

   cdef np.ndarray[DCTYPE_t, ndim=2] AH
   cdef np.ndarray[DCTYPE_t, ndim=1] x
   cdef np.ndarray[DCTYPE_t, ndim=1] r0
   cdef np.ndarray[DCTYPE_t, ndim=1] r
   cdef np.ndarray[DCTYPE_t, ndim=1] r_test   
   cdef double                       normr0
   cdef double                       norm_test
   
   cdef np.ndarray[DCTYPE_t, ndim=1] p

   cdef np.ndarray[DCTYPE_t, ndim=1] bir
   cdef np.ndarray[DCTYPE_t, ndim=1] bip
   
   cdef DCTYPE_t num
   cdef DCTYPE_t den
   cdef DCTYPE_t alpha
   cdef DCTYPE_t betha
   
   cdef int i, j, k
   cdef DCTYPE_t dot_sum  = 0
   cdef DCTYPE_t dot_sum2 = 0
   
   cdef np.ndarray[DCTYPE_t, ndim=1] Ap
   cdef np.ndarray[DCTYPE_t, ndim=1] AHbip
   
   cdef np.complex128_t* ptr
   cdef np.complex128_t* ptr2 
   
#   cdef int col
#   cdef int row
#   
   A   = A_in
   b   = b_in
   x0  = x0_in
   tol = tol_in
   itr = itr_in
   
   # Check arguments
   m = A_in.shape[0]
   n = A_in.shape[1]
   if m != n:
      raise Exception('Coefficient matrix is not square')
   if len(b) != n:
      raise Exception('Dimension mismatch between A and b')

   AH        = np.zeros((m,m), dtype=np.complex)
   x        = np.zeros(m, dtype=np.complex)
   r0       = np.zeros(m, dtype=np.complex)
   r        = np.zeros(m, dtype=np.complex)
   r_test   = np.zeros(m, dtype=np.complex)
   p        = np.zeros(m, dtype=np.complex)
   bir      = np.zeros(m, dtype=np.complex)
   bip      = np.zeros(m, dtype=np.complex)
   Ap       = np.zeros(m, dtype=np.complex)
   AHbip    = np.zeros(m, dtype=np.complex)
   
   x = x0 # make origin as starting point if no x is passed 
      
   # Calculate initial residual r = b - Ax0 and search direction p
   # EXPANDING:  r0 = b - np.dot(A, x):
   for i in range(m):
      dot_sum = b[i]
      for j in range(m):
         dot_sum = dot_sum - A[i,j]*x[j]
      r0[i] = dot_sum


   
   r[:] = r0
   # EXPANDING  normr0 = np.sqrt(np.sum(np.vdot(r,r).real)):
   dot_sum = 0
   for i in range(m):
      dot_sum = dot_sum + r[i].conjugate() * r[i]
   normr0 = dot_sum.real**0.5
   
#   if (np.euclidnorm(r) / normr0) <= tol: # break if error is less-than tolerance. 
#      # TODO: Should also test bir
#      return x
   p[:] = r
   
   # init biresidual and bidirection
   for i in range(m):
      bir[i] = r[i].conjugate()
      bip[i] = p[i].conjugate()
   

   alpha = 0
   betha = 0
   
   # Simplification: since A is Hermitian and semi-positive definite in our case, we can avoid this step...
   # EXPANDING:  AH = A.T.conjugate():
   for i in range(m):
      for j in range(m):
         AH[i,j] = A[j,i].conjugate()

   
   if itr <= 0 or itr > m:
      itr = m
   
   for k in range(itr):
      
      # Calculate common matrix-vector products
      # EXPANDING:
#      Ap = np.dot(A, p)
#      AHbip = np.dot(AH, bip)
      for i in range(m):
         dot_sum  = 0
         dot_sum2 = 0
         for j in range(m):
            dot_sum  = dot_sum  + A[i,j]  * p[j]
            dot_sum2 = dot_sum2 + AH[i,j] * bip[j]
         Ap[i]    = dot_sum
         AHbip[i] = dot_sum2

      # EXPANDING:
      # Calculate step-length parameter
#      num = np.vdot(bir, r)
#      den = np.vdot(bip, Ap)
      num = 0
      den = 0
      for i in range(m):
         num = num + bir[i].conjugate() * r[i]
         den = den + bip[i].conjugate() * Ap[i]
      alpha = num / den
   

      # EXPANDING:
#      x = x + (alpha * p)
#      r = r - alpha * Ap
      for i in range(m):
         x[i] = x[i] + alpha * p[i]    # Obtain new solution
         r[i] = r[i] - alpha * Ap[i]   # Calculate new residual and biresidual
         
      
      # EXPANDING:
#      if np.sqrt(np.sum(np.vdot(r,r).real)) / normr0 <= tol:
      dot_sum = 0
      for i in range(m):
         dot_sum = dot_sum + r[i].conjugate() * r[i]
      norm_test = dot_sum.real**0.5
      if norm_test / normr0 <= tol:
         # TODO: Should also test bir
         return x
      
      
      # EXPANDING:
#      bir = bir - (alpha.conjugate() * AHbip)
      for i in range(m):
         bir[i] = bir[i] - (alpha.conjugate() * AHbip[i])
      
      
      # EXPANDING:
      # Calculate the biconjugacy coefficient
#      num = np.vdot(AHbip, r)
#      den = np.vdot(bip, Ap)
      num = 0
      den = 0
      for i in range(m):
         num = num + AHbip[i].conjugate() * r[i]
         den = den + bip[i].conjugate() * Ap[i]
         
      betha = -1.0 * (num / den)
      
      # Obtain new search and bi-search-direction
      # EXPANDING:
#      p = r + (betha * p)
#      bip = bir + (betha.conjugate() * bip)
      for i in range(m):
         p[i] = r[i] + (betha * p[i])
         bip[i] = bir[i] + (betha.conjugate() * bip[i])
   
   return x


cpdef np.ndarray[DCTYPE_t, ndim=1] solveCholesky(np.ndarray[DCTYPE_t, ndim=2] A, np.ndarray[DCTYPE_t, ndim=1] b, int n):
   ''' Solves the Hermitian positive definite system Ax = b '''
   
   cdef np.ndarray[DCTYPE_t, ndim=2] U
   cdef np.ndarray[DCTYPE_t, ndim=2] UT
   cdef np.ndarray[DCTYPE_t, ndim=1] y
   cdef np.ndarray[DCTYPE_t, ndim=1] x
   cdef int i,j
   
   
   UT = np.zeros((n,n),dtype=CTYPE)
   
   U = cholesky(A, n)
   
   for i in range(n):
      for j in range(n):
         UT[j,i] = U[i,j].conjugate()
          
#   UT = matrixTransposedS(U, n) # TODO: Change to np.conjugatetranspose()
   
   y = forwardSolve(UT, b, n)
   x = backtrackSolve(U, y, n)
   
   return x

#cdef inline np.ndarray[DCTYPE_t, ndim=2] matrixTransposedS(np.ndarray[DCTYPE_t, ndim=2] A, int n):
#   ''' Transpose the nxn matrix A '''
#   cdef np.ndarray[DCTYPE_t, ndim=2] C = np.zeros((n,n),dtype=np.complex)
#   cdef int i,j
#   
#   for i in range(n):
#      for j in range(n):
#         C[j,i] = A[i,j].conjugate()
#         
#   return C


cpdef np.ndarray[DCTYPE_t, ndim=1] solveUHDU(np.ndarray[DCTYPE_t, ndim=2] A, np.ndarray[DCTYPE_t, ndim=1] b, int n):
   ''' Solves the Hermitian positive definite system Ax = b using U'DU decomposition'''
   
#   ???
   U,D = uhdu(A, n)
   
   #TODO: Check that the correct dimensions is cached.
   cdef np.ndarray[DCTYPE_t, ndim=2] UH = U.T.conj() #matrixTransposedS(U, n) # TODO: Change to np.conjugatetranspose()
   
   cdef np.ndarray[DCTYPE_t, ndim=1] y = forwardSolve(UH, b, n)
   cdef np.ndarray[DCTYPE_t, ndim=1] z = diagonalSolve(D, y, n)
   cdef np.ndarray[DCTYPE_t, ndim=1] x = backtrackSolve(U, z, n)
   
   return x


cpdef np.ndarray[DCTYPE_t, ndim=1] forwardSolve(np.ndarray[DCTYPE_t, ndim=2] A, np.ndarray[DCTYPE_t, ndim=1] b, int n):
   ''' Forward solve the lower triangular system Ax = b '''
   cdef np.ndarray[DCTYPE_t, ndim=1] x
   cdef int i,k
   
   x = np.zeros(n,dtype=CTYPE)
   for i in range(n):
      
      x[i] = b[i]
      
      for k in range(i):
         x[i] = x[i] - A[i,k] * x[k]
      
      x[i] = x[i] / A[i,i] 
   
   return x


cpdef np.ndarray[DCTYPE_t, ndim=1] backtrackSolve(np.ndarray[DCTYPE_t, ndim=2] A, np.ndarray[DCTYPE_t, ndim=1] b, int n):
   ''' Backtrack solve the upper triangular system Ax = b '''
   cdef np.ndarray[DCTYPE_t, ndim=1] x
   cdef int i,k
   
   x = np.zeros(n,dtype=CTYPE)
   
   for i in range(n-1, -1, -1):
      
      x[i] = b[i]
      
      for k in range(n-1, i, -1):
         x[i] = x[i] - A[i,k] * x[k]
         
      x[i] = x[i] / A[i,i]
   
   return x

cpdef np.ndarray[DCTYPE_t, ndim=1] diagonalSolve(np.ndarray[DCTYPE_t, ndim=1] A, np.ndarray[DCTYPE_t, ndim=1] b, int n):
   ''' Solve the diagonal system Ax = b. A is sparse, hence a vector '''
   cdef np.ndarray[DCTYPE_t, ndim=1] x = np.zeros(n,dtype=np.complex)
   
   for i in range(n):      
      x[i] = b[i] / A[i]
   
   return x



#def solveCholesky(A_in, b_in, n_in):
#   ''' Solves the Hermitian positive definite system Ax = b '''
#   
#   cdef np.ndarray[DCTYPE_t, ndim=2] A = A_in
#   cdef np.ndarray[DCTYPE_t, ndim=1] b = b_in
#   cdef int n = n_in
#   
#   
#   cdef np.ndarray[DCTYPE_t, ndim=2] U
#   cdef np.ndarray[DCTYPE_t, ndim=2] UT
#   cdef np.ndarray[DCTYPE_t, ndim=1] y
#   cdef np.ndarray[DCTYPE_t, ndim=1] x
#   
#   
#   UT = np.zeros((n,n),dtype=CTYPE)
#   
#   U = cholesky(A, n)
#   
#   for i in range(n):
#      for j in range(n):
#         UT[j,i] = A[i,j].conjugate()
#          
##   UT = matrixTransposedS(U, n) # TODO: Change to np.conjugatetranspose()
#   
#   y = forwardSolve(UT, b, n)
#   x = backtrackSolve(U, y, n)
#   
#   return x
#
#
#def solveUHDU(A_in, b_in, n_in):
#   ''' Solves the Hermitian positive definite system Ax = b using U'DU decomposition'''
#   
#
#   cdef np.ndarray[DCTYPE_t, ndim=2] A = A_in
#   cdef np.ndarray[DCTYPE_t, ndim=1] b = b_in
#   cdef int n = n_in
#
#   U,D = uhdu(A, n)
#   
#   #TODO: Check that the correct dimensions is cached.
#   cdef np.ndarray[DCTYPE_t, ndim=2] UH = U.T.conj() #matrixTransposedS(U, n) # TODO: Change to np.conjugatetranspose()
#   
#   cdef np.ndarray[DCTYPE_t, ndim=1] y = forwardSolve(UH, b, n)
#   cdef np.ndarray[DCTYPE_t, ndim=1] z = diagonalSolve(D, y, n)
#   cdef np.ndarray[DCTYPE_t, ndim=1] x = backtrackSolve(U, z, n)
#   
#   return x
#
#
#def forwardSolve(A_in, b_in, n_in):
#   ''' Forward solve the lower triangular system Ax = b '''
#   cdef np.ndarray[DCTYPE_t, ndim=2] A = A_in
#   cdef np.ndarray[DCTYPE_t, ndim=1] b = b_in
#   cdef int n = n_in
#   
#   cdef np.ndarray[DCTYPE_t, ndim=1] x
#   cdef int i,k
#   
#   x = np.zeros(n,dtype=CTYPE)
#   for i in range(n):
#      
#      x[i] = b[i]
#      
#      for k in range(i):
#         x[i] = x[i] - A[i,k] * x[k]
#      
#      x[i] = x[i] / A[i,i] 
#   
#   return x
#
#
#def backtrackSolve(A_in, b_in, n_in):
#   ''' Backtrack solve the upper triangular system Ax = b '''
#   cdef np.ndarray[DCTYPE_t, ndim=2] A = A_in
#   cdef np.ndarray[DCTYPE_t, ndim=1] b = b_in
#   cdef int n = n_in
#   
#   cdef np.ndarray[DCTYPE_t, ndim=1] x
#   cdef int i,k
#   
#   x = np.zeros(n,dtype=CTYPE)
#   
#   for i in range(n-1, -1, -1):
#      
#      x[i] = b[i]
#      
#      for k in range(n-1, i, -1):
#         x[i] = x[i] - A[i,k] * x[k]
#         
#      x[i] = x[i] / A[i,i]
#   
#   return x
#
#def diagonalSolve(A_in, b_in, n_in):
#   ''' Solve the diagonal system Ax = b. A is sparse, hence a vector '''
#   cdef np.ndarray[DCTYPE_t, ndim=2] A = A_in
#   cdef np.ndarray[DCTYPE_t, ndim=1] b = b_in
#   cdef int n = n_in
#   
#   cdef np.ndarray[DCTYPE_t, ndim=1] x = np.zeros(n,dtype=np.complex)
#   
#   for i in range(n):      
#      x[i] = b[i] / A[i]
#   
#   return x
