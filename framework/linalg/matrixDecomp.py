'''
Created on 20. jan. 2012

@author: jpaasen_adm
'''

from framework.mynumpy import zeros, eye

def cholesky(A, n):
   ''' Calculates the Cholesky decomposition of the Hermitian matrix A such that U is upper triangular and U'*U = A '''
   
   U = zeros([n, n], A.dtype)
   
   for i in range(n):
      
      upperColSum = 0
      for k in range(i):
         upperColSum += U[k, i] * (U[k, i]).conjugate() 
            
      #U[i, i] = cmath.sqrt(A[i, i] - upperColSum)
      U[i, i] = (A[i, i] - upperColSum)**0.5
          
      for j in range(i+1,n): 
            
         upperColSum = 0
         for k in range(i):
            upperColSum += U[k, i].conjugate() * U[k, j] 
            
         U[i, j] = (A[i, j] - upperColSum) / U[i, i]
   
   return U


def choleskyInPlace(A, n, clearUpperTriangel = True):
   ''' Calculates the Cholesky decomposition of A inplace '''
   
   ''' The lower triangular L of A will be returned with values such that LL' = A '''
   ''' L' equals U from the cholesky algorithm above '''
   
   for k in range(n):
      A[k, k] = A[k, k]**0.5
      
      for i in range(k+1,n):
         A[i, k] /= A[k, k]
      
      for j in range(k+1,n):
         for i in range(j,n):
            A[i, j] -= A[i, k]*A[j, k].conjugate()

   if clearUpperTriangel:      
      # zero out rest elements 
      for i in range(n):
         for j in range(i+1,n):
            A[i, j] = 0;
   
   return A
   

def uhdu(A, n):
   ''' 
   Calculates the UDUH decomposition of the Hermitian matrix A 
   such that U is unit upper triangular, D is diagonal and UDU'=A 
   (' = H = symmetric conjugated)
   
   Now we avoid using the complex sqrt by instead introducing two complex add
   
   Returns [U D]
   '''
   U = eye(n, dtype=A.dtype)
   D = zeros(n, dtype=A.dtype)
   
   for i in range(n):
      
      upperColSum = 0
      for k in range(i):
         upperColSum += U[k, i] * U[k, i].conjugate() * D[k]
            
      D[i] = A[i, i] - upperColSum
          
      for j in range(i+1,n): 
            
         upperColSum = 0
         for k in range(i):
            upperColSum += U[k, i].conjugate() * U[k, j] * D[k]
            
         U[i, j] = (A[i, j] - upperColSum) / D[i]
   
   return [U, D]

def uhduGPUProto(A, n):
   ''' 
   Calculates the UDUH decomposition of the Hermitian matrix A 
   such that U is unit upper triangular, D is diagonal and UDU'=A 
   (' = H = symmetric conjugated)
   
   Now we avoid using the complex sqrt by instead introducing two complex add
   
   Returns [U D]
   
   prototype CPU-code for how uhdu-composition should be done on the GPU
   '''
   U = eye([n, n], A.dtype)
   D = zeros([n,1], A.dtype)
   
   upperColSum = zeros([n,1], A.dtype)   # shared column sum buffer
   Ai = zeros([n,1], A.dtype)            # shared A row buffer
   
   for i in range(n):
      
      # read one row into "shared" memory
      for k in range(n):
         Ai[k, 0] = A[i, k]
      
      upperColSum = 0
      for k in range(i):
         upperColSum += U[k, i] * U[k, i].conjugate() * D[k, 0]
            
      D[i, 0] = A[i, i] - upperColSum
          
      for j in range(i+1,n): 
            
         upperColSum = 0
         for k in range(i):
            upperColSum += U[k, i].conjugate() * U[k, j] * D[k, 0]
            
         U[i, j] = (A[i, j] - upperColSum) / D[i, 0]
   
   return [U, D]


