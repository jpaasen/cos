
from framework.mynumpy import db, abs

import framework.mynumpy as np

from framework.System import System




def testMVDRKernelPerformance(M=32,L=16,K=0):
#   c = 12345*6789
#   print 'The result of 12345 x 6789 :', c
#   return c
   
   import framework.beamformer.capon.getCaponCUDA as getCaponCUDA
#   import scipy.interpolate as interpolate
   
   s = System('Holmengraa')
   
   r = [3000,9000]
   
   #Ny, Nx, M = s.data.Xd.shape
   
  
   Xd = s.data.Xd[r[0]:r[1],:,0:M]
   Nt = s.data.Nt[r[0]:r[1]]
   
   ext=(0,62.76,Nt.max(),Nt.min())
      
   fs = s.data.fs

   Ny, Nx, M = Xd.shape
   
   das = Xd.sum(2)
 
   d = 0.01
#   L = 16
#   K = 0
   V = np.array([])

#   from time import sleep
#   sleep(7)

   Xd = Xd[:500,:100].copy()
   
   print "  Running getCaponCUDA.getCaponCUDAPy(Xd[100x10xM=%d], d, L=%d, K, V, False, False)"%(M,L) 
   mvdr1 = getCaponCUDA.getCaponCUDAPy(Xd, d, L, K, V, False, False)

if __name__ == "__main__":
    testMVDRKernelPerformance()
