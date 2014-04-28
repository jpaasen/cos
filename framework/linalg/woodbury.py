'''
Created on 20. jan. 2012

@author: jpaasen_adm
'''

import framework.mynumpy as mynp


def iterativeBuildUpRinv(Ainv, X, M, L):
   '''
   Update Ainv with the data in X, using sub-arrays of length L.
   Remember to normalize X before calling this function.
   '''
   newAinv = mynp.array(Ainv)
   
   K = M - L + 1;
   
   for l in range(K):
      newAinv = shermanMorrisonUp(X[l:l+L], newAinv, L)
         
   return newAinv

def iterativeBuildDownRinv(Ainv, X, M, L):
   '''
   Down grade Ainv with the data in X, using sub-arrays of length L.
   Remember to normalize X before calling this function.
   '''
 
   newAinv = mynp.array(Ainv)
 
   K = M - L + 1;
   
   for l in range(K):
      newAinv = shermanMorrisonDown(X[l:l+L], newAinv, L)
      
   return newAinv
         

def iterativeBuildRinv(X, M, L, Yavg, d):
   '''
   Builds Rinv from the data area X
   using M elements, L long sub arrays, 2*Yavg+1 time samples and diagonal loading d.
   X needs to be of size (2*Yavg+1, M).
   
   The inverse is normalized with 1 / ((2*Yavg+1) * (M-l+1))
   '''
   diagonalLoadingFactor = 1
   if d > 0:  # if d is 0, use 1. Should use a smaller value, but it is not possible due to poor conditioning
      diagonalLoadingFactor = 1 / d
   
   Rinv = (diagonalLoadingFactor * mynp.eye(L));
   
   N = 2*Yavg + 1;
   K = M - L + 1;
   
   for n in range(N):
      Rinv = iterativeBuildUpRinv(Rinv, X[n], M, L)
      
   Rinv = (N * K) * Rinv # normalize Rinv
   
   return Rinv; 

def woodbury(X, Ainv, m, up = True):
   '''
   Does a rank-k adjustment of Ainv using the woodbury matrix identity
   (A + XX^H)^(-1) = A^(-1) - (A^(-1)X ( I + X^(H)A^(-1)X )^(-1)X^(H)A^(-1))
   If up == True: A rank update is performed, 
   otherwise we do a rank degrade
   
   NB! Not finished
   ''' 
   factor = 1
   if up != True:
      factor = -1
   
   XH = mynp.conjugatetranspose(X)
   XHAinv = mynp.dot(XH, Ainv)
   AinvX = mynp.dot(Ainv, XH)
   XHAinvX = mynp.dot(XHAinv, X)
   
   IXHAinvX = mynp.eye(m) + (factor * XHAinvX)
   # TODO: inv_IXHAinvX = we need to invert IXHAinvX!!!
   
   AinvX_IXHAinvX_XHAinv = mynp.dot(mynp.dot(AinvX, IXHAinvX), XHAinv)
   
   newAinv = Ainv + (-factor * AinvX_IXHAinvX_XHAinv)
   
   return newAinv

def shermanMorrison(x, Ainv, m, up = True):
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
   Ainvx = mynp.dot(Ainv, x)
   xHAinv = mynp.dot(xH, Ainv)
   xHAinvx = mynp.dot(xH, mynp.dot(Ainv, x)) # memory layout impose this product rather than reusing xHAinv (the result here is a scalar)
   
   numerator = mynp.outer(Ainvx, xHAinv)
   denominator = 1 + factor * xHAinvx
   if abs(denominator) == 0:
      raise Exception('Denominator is 0 in Sherman Morrison formula')
   
   return Ainv - ( (factor/denominator) * numerator )


def shermanMorrisonUp(x, Ainv, m):
   ''' Does a rank-1 upgrade of Ainv using the Sherman-Morrison formula '''
   return shermanMorrison(x, Ainv, m)
         

def shermanMorrisonDown(x, Ainv, m):
   ''' Does a rank-1 downgrade of Ainv using the Sherman-Morrison formula '''
   return shermanMorrison(x, Ainv, m, False)
