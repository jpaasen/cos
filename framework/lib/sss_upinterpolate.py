
import framework.mynumpy as np
import scipy.interpolate as interpolate
from scipy import signal

def upinterpolate(img):
   #
   # img    Ny x Nx   
   
   Ny,Nx = img.shape
   
   Kx = 2 # Must be even!!!
   
   new_img = np.zeros((Ny,Kx*Nx-Kx),dtype=img.dtype)
   
   for i in range(Kx):
      new_img[:,i::Kx] = img[:,0:-1] + (i+0.5)*(img[:,1:] - img[:,0:-1])/Kx
#   
#   new_img[:,0::Kx] = img[:,0:-1] + a*(img[:,1:]   - img[:,0:-1])
#   new_img[:,1::Kx] = img[:,1:]   + 0.75*(img[:,0:-1] - img[:,1:]  )
   
   return new_img   
   
   