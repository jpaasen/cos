from framework.mynumpy import pi
import framework.mynumpy as np

from framework.lib.mkl.mklUnivariateSplineC import mklcUnivariateSplineC


from scipy import interpolate

def tdib( mfdata, mtaxe, x_r, D, c, fc, xarr, yarr ):
   
   # Check if mfdata is 3D. Multiple banks is not supported.
   mfz = mfdata.shape
   if mfz.__len__() > 2:
      print 'Several banks not supported. Only processing first bank'
      mfdata = mfdata[:,:,1]
      mfz = mfdata.shape
   
   
   # Estimate the sizes
   n_hydros = mfz[0]
   n_x = xarr.shape[0]
   n_y = yarr.shape[1]
   
   # Define some useful variables
   jtopifc = 2j*pi*fc
   inv_c   = 1./c
   
   # Choose center of PCA as origin in x and center of PCA as origin in z
   xx = np.mean(x_r)/2
   zz = np.mean(D)/2
   
   # For all beams 
   image = np.zeros((n_x,n_y),dtype=complex)
   for nx in range(n_x):
      
      # Range from transmitter to slant-range beam
      r_t = np.sqrt( ( xx + xarr[nx] )**2 + yarr[nx]**2 + zz**2 )
      
      # For all hydrophones
      for n in range(n_hydros):
         
         # Range from slant-range beam to receiver
         r_r = np.sqrt( ( xx + xarr[nx] - x_r[n] )**2 + yarr[nx]**2 + ( zz - D[n] )**2 )
   
         # Travel time for all pixels in this beam and this hydrophome
         tipos = ( r_t + r_r ) * inv_c
   
         # Select ping data
         
         # Interpolate into correct time for all pixels in this beam and this hydrophone
#         tmp2 = interp1( mtaxe, tmp, tipos, 'linear' )
              
         tmp = mfdata[n]
                 
         tmp2_re = interpolate.interp1d(mtaxe,tmp.real,kind='linear',bounds_error=False)(tipos)
         tmp2_im = interpolate.interp1d(mtaxe,tmp.imag,kind='linear',bounds_error=False)(tipos)
         tmp2 = tmp2_re + 1j*tmp2_im
         
         # Mix to carrier
         tmp3 = tmp2 * np.exp( jtopifc * tipos )
         
         # Sum with all the other hydrophones
         image[nx] = image[nx] + tmp3
         
      
   
   # Scale output with the number of hydrophones
   return image.T.copy() * (1.0/n_hydros)