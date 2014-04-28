'''
Created on 20. jan. 2012

@author: jpaasen_adm
'''

import framework.mynumpy as np
import matrixDecomp as md

def solveBiCG(A, b, x0, tol, itr):
   
   ''' Solves the complex linear system Ax = b using the complex-biconjugate-gradient method
      
      Arguments:
      A   -- Coefficient matrix
      b   -- Right-side vector
      x0  -- Initial solution (default: 0-vector)
      tol -- Stop solver when error is less than tol (default: 0)
      itr -- Stop solver after itr iterations -- default
   '''
   
   # Check arguments
   m,n = len(A),len(A[0])
   if m != n:
      raise Exception('Coefficient matrix is not square')
   if len(b) != n:
      raise Exception('Dimension mismatch between A and b')
   
   x = x0 # make origin as starting point if no x is passed 
      
   # Calculate initial residual r = b - Ax0 and search direction p
   r0 = b - np.dot(A, x)
   r = r0
   normr0 = np.euclidnorm(r0)
   
   if (np.euclidnorm(r) / normr0) <= tol: # break if error is less-than tolerance. 
      # TODO: Should also test bir
      return x
   p = np.array(r)
   
   # init biresidual and bidirection
   bir = r.conjugate()
   bip = p.conjugate()
   
   numerator,denominator = 0,0
   alpha,betha = 0,0
   AH = np.conjugatetranspose(A) # Simplification: since A is Hermitian and semi-positive definite in our case, we can avoid this step...
   
   if itr <= 0 or itr > m:
      itr = m
   
   for i in range(itr):
      
      # Calculate common matrix-vector products
      Ap = np.dot(A, p)
      AHbip = np.dot(AH, bip)
      
      # Calculate step-length parameter
      numerator = np.vdot(bir, r)
      denominator = np.vdot(bip, Ap)
      alpha = numerator / denominator
      
      # Obtain new solution
      x = x + (alpha * p)
      
      # Calculate new residual and biresidual
#      r = r - (alpha * Ap)
      
      for i in range(m):
         r[i] = r[i] - alpha * Ap[i]   # Calculate new residual and biresidual
      
      if np.euclidnorm(r) / normr0 <= tol:
         # TODO: Should also test bir
         return x
      bir = bir - (alpha.conjugate() * AHbip)
      
      # Calculate the biconjugacy coefficient
      numerator = np.vdot(AHbip, r)
      denominator = np.vdot(bip, Ap)
      betha = -1.0 * (numerator / denominator)
      
      # Obtain new search and bi-search-direction
      p = r + (betha * p)
      bip = bir + (betha.conjugate() * bip)
   
   return x

def solveCholesky(A, b, n):
   ''' Solves the Hermitian positive definite system Ax = b using cholesky decomposition'''
   
   U = md.cholesky(A, n)

   UT = np.conjugatetranspose(U)
   
   y = forwardSolve(UT, b, n)
   x = backtrackSolve(U, y, n)
   
   return x

def solveCholeskyCpp(A, b, n):
   ''' Solves the Hermitian positive definite system Ax = b using cholesky decomposition'''
   
   U = md.cholesky(A, n)

   UT = np.conjugatetranspose(U)
   
   y = forwardSolve(UT, b, n)
   x = backtrackSolve(U, y, n)

def solveUHDU(A, b, n):
   ''' Solves the Hermitian positive definite system Ax = b using U'DU decomposition'''
   
   UD = md.uhdu(A, n);
   
   UH = np.conjugatetranspose(UD[0])
   
   y = forwardSolve(UH, b, n)
   z = diagonalSolve(UD[1], y, n)
   x = backtrackSolve(UD[0], z, n)
   
   return x


' Under follows methods for solving triangular and diagonal systems '

def forwardSolve(A, b, n):
   ''' Forward solve the lower triangular system Ax = b '''
   x = np.zeros(n, A.dtype)
   
   for i in range(n):
      
      x[i] = b[i]
      
      for k in range(i):
         x[i] -= A[i, k] * x[k]
      
      x[i] /= A[i, i] 
   
   return x


def backtrackSolve(A, b, n):
   ''' Backtrack solve the upper triangular system Ax = b '''
   x = np.zeros(n, A.dtype)
   
   for i in reversed(range(n)):
      
      x[i] = b[i]
      
      for k in reversed(range(i+1, n)):
         x[i] -= A[i, k] * x[k]
         
      x[i] /= A[i, i]
   
   return x

def diagonalSolve(A, b, n):
   ''' Solve the diagonal system Ax = b, A is sparse, hence a vector '''
   x = np.zeros(n, A.dtype)
   
   for i in range(n):      
      x[i] = b[i] / A[i]
   
   return x
