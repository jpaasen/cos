
#include "mklvcExp.h"
#include "mkl.h"
#include "mkl_df_defines.h"
#include "mkl_service.h"
#include "stdio.h"


//template <class T>
typedef float T;

int mklvcExp( const int n, const MKL_Complex8 *a, MKL_Complex8 *y )
//int Nx,  T *x, T *y,
//                        int Nyi, T *xi, T *yi,
//                        int kind, bool xi_uniform )
{
   mkl_set_num_threads(9);

   vcExp(n,a,y);

   return 0;
}

