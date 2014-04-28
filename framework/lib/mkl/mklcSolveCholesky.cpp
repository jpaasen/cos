
#include "mklcSolveCholesky.h"
#include "mkl.h"
#include "stdio.h"
#include "math.h"
//#include <Python.h>
//#include <numpy/arrayobject.h>


#define N 10                      /* Size of partition, number of breakpoints */
#define NSITE 91                  /* Number of interpolation sites */

#define SPLINE_ORDER DF_PP_CUBIC    /* A cubic spline to construct */

#define LAPACK_ROW_MAJOR 101
#define LAPACK_COL_MAJOR 102


void printStatus(const char *desc, int status) {

#ifdef DEBUG
   if( status == 0 ) {
      printf( "%s : Operation completed successfully.\n", desc );
   }
   else if( status < 0 ) {
      printf( "%s : ERROR: Problem with parameter %d.\n", desc, abs(status) );
   }
   else {
      printf( "%s : ERROR: Matrix is not positive definite!!!.\n", desc );
   }
#endif

}

int mlkcSolveCholesky(Complex<double> *a_in, Complex<double> *b_in, int order)
{

   // Factorisation/solver parameters:
   char             uplo            = 'L';  // Upper triangular part is stored
   int              n               = order;    // Order of the matrix
   int              lda             = order;    // TODO: Leading dimension of a
   int              ldb             = 1;    // TODO: Leading dimension of b
   double           tol             = -1.0; // Pivot tolerance
                      // Upper triagonal matrix
//                    *b;                     // Result matrix
   int              nrhs            = 1;    // Number of right-hand sides
   int              matrix_order    = LAPACK_ROW_MAJOR;

//   MKL_Complex16    a[4];
//   MKL_Complex16    b[2];

   double *work = new double[2*n];

   int              piv;                    // Pivot???
   int              rank;                   // Rank (determined by number of steps)

   MKL_Complex16 *a = (MKL_Complex16*)a_in;
   MKL_Complex16 *b = (MKL_Complex16*)b_in;

//   double ar[4] = {3,5,5,14};
//   double ai[4] = {0,1,-1,0};
//
//   for( int i=0; i<n*n; ++i){
//      a[i].real = ar[i];
//      a[i].imag = ai[i];
//   }
//
//   double br[2] = {1,2};
//   double bi[2] = {1,1};
//   for( int i=0; i<2; ++i){
//      b[i].real = br[i];
//      b[i].imag = bi[i];
//   }


   int info;

//   zpstf2( &uplo, &n, a, &lda, &piv, &rank, &tol, work, &info );


   info = LAPACKE_zpotrf( matrix_order, uplo, n, a, lda );

   printStatus("zpotrf()", info);

#ifdef DEBUG
   for( int i=0; i<n; ++i){
      for( int j=0; j<n; ++j){
         printf("%2.4f %2.4f   ", a[i*2+j].real, a[i*2+j].imag );
      }
      printf("\n");
   }
#endif

   info = LAPACKE_zpotrs( matrix_order, uplo, n, nrhs, (const MKL_Complex16*)a, lda, b, ldb );

   printStatus("zpotrs()", info);

#ifdef DEBUG
   for( int i=0; i<n; ++i){
      for( int j=0; j<n; ++j){
         printf("%2.4f %2.4f   ", b[i*2+j].real, b[i*2+j].imag );
      }
      printf("\n");
   }
#endif

   return 0;
}
