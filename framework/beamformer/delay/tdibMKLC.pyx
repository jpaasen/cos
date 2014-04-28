from framework.mynumpy import pi
import framework.mynumpy as np



from scipy import interpolate
from framework.lib.mkl.mklUnivariateSplineC import mklcUnivariateSplineC

FTYPE_s = 'float64' 
CTYPE_s = 'complex128'
FTYPE=np.float64
CTYPE=np.complex128

###CYTHON_MODE
cimport numpy as np
ctypedef np.float64_t FTYPE_t
ctypedef np.complex128_t CTYPE_t
from libcpp cimport bool

cimport numpy as np
cpdef tdib( mfdata_in, mtaxe_in, x_r_in, D_in, c_in, fc_in, xarr_in, yarr_in ):

   cdef int n_hydros, n_y, n_x,  n, i, nx#, n_y, i
   cdef FTYPE_t c, fc, inv_c, xx, zz, dtmp, exp, dz
   cdef np.ndarray[CTYPE_t, ndim=1] tmp, tmp2, tmp3
   cdef np.ndarray[FTYPE_t, ndim=1] mtaxe, x_r, D, r_t, r_r, tipos
   cdef CTYPE_t ctmp, ctmp2
#   cdef np.complex64_t ctmp, ctmp2, 
#   cdef np.ndarray[np.complex128_t, ndim=1] hmm
   
#   cdef np.ndarray[np.complex128_t, ndim=2] 
   cdef np.ndarray[CTYPE_t, ndim=2] mfdata, image
   cdef np.ndarray[FTYPE_t, ndim=2] xarr, yarr #, image, 
#   cdef np.ndarray[np.complex128_t, ndim=2] image
   
#   tmp2  = np.zeros(n_y,dtype=np.complex64)
#   tmp3  = np.zeros(n_y,dtype=np.complex64)
   
   ctmp  = 0
   ctmp2 = 0
   exp   = np.e
   
###CYTHON_MODE_END§
   """PYTHON_MODE
#@profile
def tdib( mfdata_in, mtaxe_in, x_r_in, D_in, c_in, fc_in, xarr_in, yarr_in ):
PYTHON_MODE_END"""

#   is_uniform = False
#
#   # Check if mfdata is 3D. Multiple banks is not supported.
#   mfz = mfdata_in.shape
#   if mfz.__len__() > 2:
#      print 'Several banks not supported. Only processing first bank'
#      mfdata = mfdata[:,:,1]
#      mfz = mfdata_in.shape
   
   mfdata = mfdata_in.astype(CTYPE_s)
   mtaxe = mtaxe_in.astype(FTYPE_s)
   x_r   = x_r_in.astype(FTYPE_s)
   D     = D_in.astype(FTYPE_s)
   c     = c_in
   fc    = fc_in
   xarr  = xarr_in.astype(FTYPE_s)
   yarr  = yarr_in.astype(FTYPE_s)
   
   
   # Estimate the sizes
   n_hydros = mfdata.shape[0]
   n_x = xarr_in.shape[0]
   n_y = yarr_in.shape[1]
   

   tipos = np.zeros(n_y,dtype=FTYPE)
   r_t   = np.zeros(n_y,dtype=FTYPE)
   r_r   = np.zeros(n_y,dtype=FTYPE)
   tmp   = np.zeros(n_y,dtype=CTYPE)
   tmp2   = np.zeros(n_y,dtype=CTYPE)
   tmp3   = np.zeros(n_y,dtype=CTYPE)
   
   # Define some useful variables
   jtopifc = 2j*np.pi*fc
   inv_c   = 1./c
   
   # Choose center of PCA as origin in x and center of PCA as origin in z
   xx = np.mean(x_r)/2
   zz = np.mean(D)/2
   
   # For all beams 
   image = np.zeros((n_x,n_y),dtype=CTYPE)
   for nx in range(n_x):
      
      # Range from transmitter to slant-range beam
      ###CYTHON_MODE
      for i in range(n_y):
         r_t[i] = ( ( xx + xarr[nx,i] )**2 + yarr[nx,i]**2 + zz**2 )**0.5
      ###CYTHON_MODE_END§
      """PYTHON_MODE
      r_t = ( ( xx + xarr[nx] )**2 + yarr[nx]**2 + zz**2 )**0.5
      PYTHON_MODE_END"""

      # For all hydrophones
      for n in range(n_hydros):
         
         ##CYTHON_MODE
#         for i in range(n_y):
#            r_r[i] = ( ( xx + xarr[nx,i] - x_r[n] )**2 + yarr[nx,i]**2 + ( zz - D[n] )**2 )**0.5
         ###CYTHON_MODE_END§
         """PYTHON_MODE
         r_r = np.sqrt( ( xx + xarr[nx] - x_r[n] )**2 + yarr[nx]**2 + ( zz - D[n] )**2 )
         PYTHON_MODE_END"""
         r_r = np.sqrt( ( xx + xarr[nx] - x_r[n] )**2 + yarr[nx]**2 + ( zz - D[n] )**2 )
         # Range from slant-range beam to receiver
         
   
         # Travel time for all pixels in this beam and this hydrophome
         ##CYTHON_MODE
#         for i in range(n_y):
#            dtmp = r_t[i] + r_r[i]
#            tipos[i] = dtmp * inv_c
#         ###CYTHON_MODE_END§
#         """PYTHON_MODE
#         tipos = ( r_t + r_r ) * inv_c
#         PYTHON_MODE_END"""
         tipos = ( r_t + r_r ) * inv_c
   
         # Select ping data            
         tmp = mfdata[n]
                 
         tmp2 = mklcUnivariateSplineC( mtaxe, tmp, tipos, 0, False ) 
#         tmp2_re = interpolate.interp1d(mtaxe,tmp.real,kind='linear',bounds_error=False)(tipos)
#         tmp2_im = interpolate.interp1d(mtaxe,tmp.imag,kind='linear',bounds_error=False)(tipos)
#         tmp2 = tmp2_re + 1j*tmp2_im
         
#         # Mix to carrier
#         ###CYTHON_MODE
#         for i in range(n_y):
#            ctmp = jtopifc * tipos[i]
#            ctmp2 =  exp**ctmp
#            tmp3[i]= tmp2[i] * ctmp2
#         ###CYTHON_MODE_END§
#         """PYTHON_MODE
#         tmp3 = tmp2 * np.exp( jtopifc * tipos )
#         PYTHON_MODE_END"""
#         hmm = <double complex> jtopifc * tipos
#         tmp = np.exp(hmm)
#         tmp3= tmp2 * ctmp2
         
         tmp3 = tmp2 * np.exp( jtopifc * tipos )
#         
#         # Sum with all the other hydrophones
#         for i in range(n_y):
#            image[nx,i] = image[nx,i] + tmp3[i]
         image[nx] = image[nx] + tmp3
      
   
   # Scale output with the number of hydrophones
   return image.T.copy() * (1.0/n_hydros)
