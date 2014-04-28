# Add the framework to the Python path
import sys, os
PHDCODE_ROOT = os.environ['PHDCODE_ROOT']
sys.path.append("%s/python"%PHDCODE_ROOT)
#sys.path.append(os.getcwd()+"/..")

from framework.Window import Rect, Kaiser, Trig
from framework.Coordinate import Coordinate

import framework.mynumpy as np
from framework.mynumpy import pi, abs, arcsin, dot, zeros, mean, db
from numpy.fft import fft, ifft
#from gfuncs import abs

from framework.System import System
from .. import getCaponC as getCaponC



def start(s):
   

   PROFILE = False

   Xd_i= Xd[0,:,:,:]

   res = getCaponC.getCapon(Xd_i, regCoef, L, nTimeAverage, V, doForwardBackwardAveraging, verbose)
        
   img_capon_ampl = res[0]
   
 
   # Steering angles:
   M = s['M'][0]
   d = s['d'][0]
   c = s['c'][0]
   fc = s['fc'][0]
   D = M*d
   lmda = float(c)/fc
     

   # Get the relevant parameters. Todo: Make system.open() for this
   Xd = s.data.Xd
   Ni = Xd.shape[0]     # No. realisations
   N  = Xd.shape[1]     # No. time samples (y-coordinates)
   Ny = N               # No. y-coordinates in image
   Nx = Xd.shape[2]     # No. x-coordinates in image
   Nm = Xd.shape[3]     # No. channels
 
  
   # Capon parameters
   regCoef        = 1.0/100         # Diagonal loading
   L              = 16              # Subarray size <= N/2 (e.g. 16)
   nTimeAverage   = 1               # Capon range-average window (+- this value)
   V              = np.array([])    # Subspace matrix
   doForwardBackwardAveraging = False
   verbose        = False
   
   Xd_i= Xd[0,:,:,:]

   res = getCaponC.getCapon(Xd_i, regCoef, L, nTimeAverage, V, doForwardBackwardAveraging, verbose)
        
   img_capon_ampl = res[0]
   
      

if '__IP' not in globals():
   
    
   s = System('HISAS speckle')

   start(s)

   print 'hey'
