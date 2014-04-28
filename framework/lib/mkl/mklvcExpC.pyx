
import numpy as np
import scipy as sp
cimport numpy as np

from libcpp cimport bool

cimport mklvcExp as exp

   int mklvcExp( int n, MKL_Complex8 *a, MKL_Complex8 *y )
def mklvcExp( int n_in, np.ndarray a_in, np.ndarray y_in ):

   cdef int n = n_in

   cdef np.ndarray[np.float32_t, ndim=1] a_py = a_in.astype('float32')
   
   cdef np.ndarray[np.float32_t, ndim=1] y_py = np.zeros(n, dtype=np.float32)
   
   cdef float *y_re = <float*>a_py.data
   
   exp.vcExp(n, a, y)
   
   
   
   cdef int Nx  = x_in.shape[0]
   cdef int Nyi = xi_in.shape[0]
   
   cdef np.ndarray[np.float32_t, ndim=1] x_py = x_in.astype('float32')
   
   cdef np.ndarray[np.float32_t, ndim=1] y_re_py = y_in.real.astype('float32')
   cdef np.ndarray[np.float32_t, ndim=1] y_im_py = y_in.imag.astype('float32')
   
   cdef np.ndarray[np.float32_t, ndim=1] yi_re_py = np.zeros(Nyi, dtype=np.float32)
   cdef np.ndarray[np.float32_t, ndim=1] yi_im_py = np.zeros(Nyi, dtype=np.float32)
   
   cdef float *x = [x_in[0],x_in[-1]]
#   <double*>x_py.data
   
   cdef float *y_re = <float*>y_re_py.data
   cdef float *y_im = <float*>y_im_py.data
   
   cdef float *yi_re = <float*>yi_re_py.data
   cdef float *yi_im = <float*>yi_im_py.data
   
   
   cdef np.ndarray[np.float32_t, ndim=1] xi_py
   
   if xi_uniform:
#      xi = [xi_in[0],xi_in[-1]]
      xi_py = np.array([xi_in[0],xi_in[-1]])
#      print 'uniform'
   else:
      xi_py = xi_in
#      print 'non-uniform'
      
   cdef float *xi    = <float*>xi_py.data

   
#   cdef np.ndarray[np.complex128_t, ndim=1] yi_py
#    
   us.mklUnivariateSpline(Nx, x, y_re, Nyi, xi, yi_re, kind, xi_uniform )
      
   us.mklUnivariateSpline(Nx, x, y_im, Nyi, xi, yi_im, kind, xi_uniform )
          
   yi_py = yi_re_py + 1j*yi_im_py
   
#   print yi_py

   return yi_py
#                
