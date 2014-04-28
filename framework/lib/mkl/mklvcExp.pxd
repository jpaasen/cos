from libcpp cimport bool

cdef extern from "mklvcExp.h":
   int mklvcExp( int n, MKL_Complex8 *a, MKL_Complex8 *y )
