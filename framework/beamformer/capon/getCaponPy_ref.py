
import numpy as np
import scipy as sp
#import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

#GETCAPON Implementation of the Minimum Variance / Capon filter for linear
#         arrays. Includes FB averaging, time averaging, and the use of
#         subspaces. 
#
# [imAmplitude imPower] = getCapon(dataCube, indsI, indsJ, regCoef, L, nTimeAverage, V, doForwardBackwardAveraging, verbose)
#
# dataCube     : A range x beams x array-elements image cube
# indsI, indsJ : Indices of pixels that are to be processed; indsI==0 / indsJ==0 means all ranges / beams
# regCoef      : Regularization coefficient for diagonal loading
# L            : Length of subarrays
# nTimeAverage : Includes +- this number of time lags to produce 'R' matrix
# V            : The columns span an orthogonal subspace; if V is empty ([]) than the full space is used (no subspacing)
# doForwardBackwardAveraging : Whether to do forward-backward averaging
#
# Note I: If using a subspace matrix, V, enabling FB averaging gives a
# substantial increase in the computational load.  This is because the
# reduction of the dimensionality must happen at a much later stage in the
# code. 
#
# Note II: This version assumes a=ones(L,1) in w = Ri*a/(a'*Ri*a), i.e., we
# search for the amplitude/power in the direction perpedicular to the
# linear array.
#
# Note III: Matrix inversion is done like this: Ri = pinv(R + regCoef/L*trace(R)*I);
#
# Last modified:
# 2009.08.25 - Are C. Jensen {Created the function}
# 2009.08.27 - Are C. Jensen {Robustified indsI in light of nTimeAverage use}
# 2009.09.09 - Are C. Jensen {By popular request, added the factor 1/L in the diagonal loading}
def getCapon(dataCube, indsI, indsJ, regCoef, L, nTimeAverage, V, doForwardBackwardAveraging, verbose):

   # Local Variables: verbose, ii, ar, nSubspaceDims, Ri, useSubspace, percent, g_singleSnapshot, dataCube, imPower, I, K, J, M, L, N, imAmplitude, R, V, doForwardBackwardAveraging, regCoef, a, d, i, indsJ, indsI, j, nTimeAverage, n, w
   # Function calls: isempty, squeeze, eye, trace, min, max, fprintf, transpose, pinv, length, ones, zeros, mod, rot90, conj, getCapon, sum, round, size
   ds = dataCube.shape
   N, M, K = ds[0], ds[1], ds[2]
   if indsI<=0:
      indsI = np.arange(0, N)
   
   if indsJ<=0:
      indsJ = np.arange(0, M)
    
   # Skip pixels that cannot be calculated due to time averaging
   indsI = indsI[indsI > nTimeAverage-1]
   indsI = indsI[indsI < N-nTimeAverage]
   
   a           = np.ones((L,))
   n           = nTimeAverage
   imAmplitude = np.zeros((N, M), dtype=complex)
   imPower     = np.zeros((N, M), dtype=complex)
   I           = np.eye(L)
   J           = np.rot90(I)
   
   useSubspace = V.__len__() != 0
   if useSubspace:
      nSubspaceDims = V.shape[1]
      if verbose:
         print 'Capon algorithm "subspaced" down to %d dims.' %nSubspaceDims
      
      
      I = np.eye(nSubspaceDims)
      a = np.dot(V.conj().T, a)
      # The column of ones (what we seek the 'magnitude' of) represented in the subspace
    
   
      for j in indsJ:
         for i in indsI:

         ar = dataCube[i-n:i+n+1,j,:].copy()
         ar_conj = ar.conj()
         # Array responses plus-minus n time steps
         if n == 0:
            ar = ar.T
         
         
         # Place the array responses in one (K-L+1)*(2n+1) x L matrix:
         d = np.zeros(((K-L)*(2*n+1), L), dtype=complex)#), order='F')
         for ii in xrange(0, K-L):
            d[ii*(2*n+1):(ii+1)*(2*n+1),:] = ar_conj[:,ii:ii+L]
             
         # If a subspace matrix V is given and we're _not_ using FB averaging, we
         # can use V to reduce the dimensionality of the data now:
         if useSubspace and not doForwardBackwardAveraging:
            d = np.dot(d, V)
         
#         d_pow2 = d**2
         R = np.dot( d.conj().T, d ) / ((K-L+1)*(2*n+1))

         # R contains an estimate of the covariance matrix
         # Store the sum of the current-time outputs in 'g_singleSnapshot':
         g_singleSnapshot = sum(d[n:2*n:,:]).conj().T / (K-L+1)
         
         if doForwardBackwardAveraging:
            R = 0.5 * ( R + np.dot(np.dot(J, R.T), J) )
         
         
         # If a subspace matrix V is given and we _are_ using FB averaging, we
         # have to wait until now to go to the reduced space:
         if useSubspace and doForwardBackwardAveraging:
            R = np.dot(np.dot(V.conj().T, R), V)
            g_singleSnapshot = np.dot(V.conj().T, g_singleSnapshot)
           
         
#         Ri = np.linalg.inv( R + regCoef * np.dot(np.trace(R), I) / L)
#         w = np.dot(Ri, a) / np.dot(np.dot(a.conj().T, Ri), a)

         R_ = R + regCoef * np.dot(np.trace(R), I) / L
         a = a.reshape((a.shape[0],1))
         w_num   = np.linalg.solve(R_, a)
         w_denum = np.dot(np.linalg.solve(R_.T, a.conj()).T, a)
         w = w_num  / w_denum 
         imAmplitude[i,j] = np.dot(w.conj().T, g_singleSnapshot)[0] # Note: A bit ad-hoc maybe, but uses only the current time-snapshot to calculate the output/'alpha' value
         imPower[i,j] = np.dot(np.dot(w.conj().T, R), w)[0,0]
           
      if verbose:
         percent = np.round( 100* (i-indsI.min()) / (indsI.max()-indsI.min()))
         if np.mod(i, 5) == 0:
            print ' %d\%'%percent
            
         if np.mod(i, 100) == 0:
            pass
            
        
   return [imAmplitude, imPower]
