
import numpy as np
import scipy as sp
cimport numpy as np

from libcpp cimport bool

cimport mklUnivariateSpline as us

FTYPE_s = 'float64'
FTYPE = np.float64
ctypedef np.float64_t FTYPE_t

def mklcUnivariateSplineC(np.ndarray x_in, np.ndarray y_in, np.ndarray xi_in, kind = 1, xi_uniform = True):

   cdef int Nx  = x_in.shape[0]
   cdef int Nyi = xi_in.shape[0]
   
   cdef np.ndarray[FTYPE_t, ndim=1] x_py = x_in.astype(FTYPE_s)
   
   cdef np.ndarray[FTYPE_t, ndim=1] y_re_py = y_in.real.astype(FTYPE_s)
   cdef np.ndarray[FTYPE_t, ndim=1] y_im_py = y_in.imag.astype(FTYPE_s)
   
   cdef np.ndarray[FTYPE_t, ndim=1] yi_re_py = np.zeros(Nyi, dtype=FTYPE)
   cdef np.ndarray[FTYPE_t, ndim=1] yi_im_py = np.zeros(Nyi, dtype=FTYPE)
   
   cdef FTYPE_t *x = [x_in[0],x_in[-1]]
#   <double*>x_py.data
   
   cdef FTYPE_t *y_re = <FTYPE_t*>y_re_py.data
   cdef FTYPE_t *y_im = <FTYPE_t*>y_im_py.data
   
   cdef FTYPE_t *yi_re = <FTYPE_t*>yi_re_py.data
   cdef FTYPE_t *yi_im = <FTYPE_t*>yi_im_py.data
   
   
   cdef np.ndarray[FTYPE_t, ndim=1] xi_py
   
   if xi_uniform:
#      xi = [xi_in[0],xi_in[-1]]
      xi_py = np.array([xi_in[0],xi_in[-1]])
#      print 'uniform'
   else:
      xi_py = xi_in
#      print 'non-uniform'
      
   cdef FTYPE_t *xi    = <FTYPE_t*>xi_py.data

   
#   cdef np.ndarray[np.complex128_t, ndim=1] yi_py
#    
   us.mklUnivariateSpline(Nx, x, y_re, Nyi, xi, yi_re, kind, xi_uniform )
      
   us.mklUnivariateSpline(Nx, x, y_im, Nyi, xi, yi_im, kind, xi_uniform )
          
   yi_py = yi_re_py + 1j*yi_im_py
   
#   print yi_py

   return yi_py
#                
