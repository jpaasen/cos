from libcpp cimport bool
cimport numpy as np

ctypedef np.float64_t FTYPE_t

cdef extern from "mklUnivariateSpline.h":
   int mklUnivariateSpline(int nx,  FTYPE_t *x, FTYPE_t *y,
                         int nyi, FTYPE_t *xi, FTYPE_t *yi,
                         int kind, bool xi_uniform )
   

#template <class T>
#int mklUnivariateSpline(int Nx,  T *x, T *y,
#                        int Nyi, T *xi, T *yi,
#                        int kind, bool xi_uniform )