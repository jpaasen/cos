
#include "mklUnivariateSpline.h"
#include "mkl.h"
#include "mkl_df_defines.h"
#include "mkl_service.h"
#include "stdio.h"


void printStatus(const char *desc, int status) {

#ifdef DEBUG
   switch( status ){

   //Common Status Codes
   case DF_STATUS_OK:
      printf( "%s : Operation completed successfully.\n", desc ); break;
   case DF_ERROR_NULL_TASK_DESCRIPTOR:
      printf( "%s : Data Fitting task is a NULL pointer.\n", desc ); break;
   case DF_ERROR_MEM_FAILURE:
      printf( "%s : Memory allocation failure.\n", desc ); break;
   case DF_ERROR_METHOD_NOT_SUPPORTED:
      printf( "%s : Requested method is not supported.\n", desc ); break;
   case DF_ERROR_COMP_TYPE_NOT_SUPPORTED:
      printf( "%s : Requested computation type is not supported.\n", desc ); break;

   // Data Fitting Task Creation and Initialization, and Generic Editing Operations
   case DF_ERROR_BAD_NX:
      printf( "%s : Invalid number of breakpoints.\n", desc ); break;
   case DF_ERROR_BAD_X:
      printf( "%s : Array of breakpoints is invalid.\n", desc ); break;
   case DF_ERROR_BAD_X_HINT:
      printf( "%s : Invalid hint describing the structure of the partition.\n", desc ); break;
   case DF_ERROR_BAD_NY:
      printf( "%s : Invalid dimension of vector-valued function y.\n", desc ); break;
   case DF_ERROR_BAD_Y:
      printf( "%s : Array of function values is invalid.\n", desc ); break;
   case DF_ERROR_BAD_Y_HINT:
      printf( "%s : Invalid flag describing the structure of function y\n", desc ); break;

   //Data Fitting Task-Specific Editing Operations
   case DF_ERROR_BAD_SPLINE_ORDER:
      printf( "%s : Invalid spline order.\n", desc ); break;
   case DF_ERROR_BAD_SPLINE_TYPE:
      printf( "%s : Invalid spline type.\n", desc ); break;
   case DF_ERROR_BAD_IC_TYPE:
      printf( "%s : Type of internal conditions used for spline construction is invalid.\n", desc ); break;
   case DF_ERROR_BAD_IC:
      printf( "%s : Array of internal conditions for spline construction is not defined.\n", desc ); break;
   case DF_ERROR_BAD_BC_TYPE:
      printf( "%s : Type of boundary conditions used in spline construction is invalid.\n", desc ); break;
   case DF_ERROR_BAD_BC:
      printf( "%s : Array of boundary conditions for spline construction is not defined.\n", desc ); break;
   case DF_ERROR_BAD_PP_COEFF:
      printf( "%s : Array of piecewise polynomial spline coefficients is not defined.\n", desc ); break;
   case DF_ERROR_BAD_PP_COEFF_HINT:
      printf( "%s : Invalid flag describing the structure of the piecewise polynomial spline coefficients.\n", desc ); break;
   case DF_ERROR_BAD_PERIODIC_VAL:
      printf( "%s : Function values at the endpoints of the interpolation interval are not equal as required in periodic boundary conditions.\n", desc ); break;
   case DF_ERROR_BAD_DATA_ATTR:
      printf( "%s : Invalid attribute of the pointer to be set or modified in Data Fitting task descriptor with the df? editidxptr task editor.\n", desc ); break;
   case DF_ERROR_BAD_DATA_IDX:
      printf( "%s : Index of the pointer to be set or modified in the Data Fitting task descriptor with the df?editidxptr task editor is out of the pre-defined range.\n", desc ); break;


   // Data Fitting Computation Operations
   case DF_ERROR_BAD_NSITE:
      printf( "%s : Invalid number of interpolation sites.\n", desc ); break;
   case DF_ERROR_BAD_SITE:
      printf( "%s : Array of interpolation sites is not defined.\n", desc ); break;
   case DF_ERROR_BAD_SITE_HINT:
      printf( "%s : Invalid flag describing the structure of interpolation sites.\n", desc ); break;
   case DF_ERROR_BAD_NDORDER:
      printf( "%s : Invalid size of the array defining derivative orders to be computed at interpolation sites.\n", desc ); break;
   case DF_ERROR_BAD_DORDER:
      printf( "%s : Array defining derivative orders to be computed at interpolation sites is not defined.\n", desc ); break;
   case DF_ERROR_BAD_DATA_HINT:
      printf( "%s : Invalid flag providing additional information about partition or interpolation sites.\n", desc ); break;
   case DF_ERROR_BAD_INTERP:
      printf( "%s : Array of spline-based interpolation results is not defined.\n", desc ); break;
   case DF_ERROR_BAD_INTERP_HINT:
      printf( "%s : Invalid flag defining the structure of spline-based interpolation results.\n", desc ); break;
   case DF_ERROR_BAD_CELL_IDX:
      printf( "%s : Array of indices of partition cells containing interpolation sites is not defined.\n", desc ); break;
   case DF_ERROR_BAD_NLIM:
      printf( "%s : Invalid size of arrays containing integration limits.\n", desc ); break;
   case DF_ERROR_BAD_LLIM:
      printf( "%s : Array of the left-side integration limits is not defined.\n", desc ); break;
   case DF_ERROR_BAD_RLIM:
      printf( "%s : Array of the right-side integration limits is not defined.\n", desc ); break;
   case DF_ERROR_BAD_INTEGR:
      printf( "%s : Array of spline-based integration results is not defined.\n", desc ); break;
   case DF_ERROR_BAD_INTEGR_HINT:
      printf( "%s : Invalid flag providing the structure of the array of spline-based integration results.\n", desc ); break;
   case DF_ERROR_BAD_LOOKUP_INTERP_SITE:
      printf( "%s : Bad site provided for interpolation with look-up interpolator.\n", desc ); break;

   // Fallback
   default:
      printf( "%s : ERROR (unknown)\n", desc ); break;
   }
#endif

}

//template <class T>
typedef double T;

int mklUnivariateSpline(int Nx,  T *x, T *y,
                        int Nyi, T *xi, T *yi,
                        int kind, bool xi_uniform )
{
   mkl_set_num_threads(9);
   int status;       /* Status of a Data Fitting operation */
   DFTaskPtr task;   /* Data Fitting operations are task based */

   /* Parameters describing the partition */
   MKL_INT xhint;       /* Additional information about the structure of breakpoints */

   /* Parameters describing the function */
   MKL_INT Ny;          /* Function dimension */
   MKL_INT yhint;       /* Additional information about the function */

   /* Parameters describing the spline */
   MKL_INT s_order;     /* Spline order */
   MKL_INT s_type;      /* Spline type */
   MKL_INT ic_type;     /* Type of internal conditions */
   T* ic;         /* Array of internal conditions */
   MKL_INT bc_type;     /* Type of boundary conditions */
   T* bc;         /* Array of boundary conditions */


   T *scoeff;  /* Array of spline coefficients */
   MKL_INT scoeffhint;           /* Additional information about the coefficients */

   /* Parameters describing interpolation computations */
   MKL_INT xihint;    /* Additional information about the structure of interpolation sites */

   MKL_INT ndorder, *dorder; /* Parameters defining the type of interpolation */
   T* datahint;    /* Additional information on partition and interpolation sites */
//   T *r = new T[Nyi];     /* Array of interpolation results */
//   for(int i=0; i<Nyi; ++i)
//      r[i] = 0;
   MKL_INT rhint;      /* Additional information on the structure of the results */
   MKL_INT* cell;       /* Array of cell indices */

   if( kind == 0 ) {
      scoeff = new T[(Nx-1)* DF_PP_LINEAR];   /* A linear spline to construct */
   }
   else {
      scoeff = new T[(Nx-1)* DF_PP_CUBIC];    /* A cubic spline to construct */
   }

   xhint = DF_UNIFORM_PARTITION; /* The partition is uniform. */

   /* Initialize the function */
   Ny = 1;              /* The function is scalar. */
   yhint = DF_NO_HINT;  /* No additional information about the function is provided. */

   /* Create a Data Fitting task */
   status = dfdNewTask1D( &task, Nx, x, xhint, Ny, y, yhint );

   /* Check the Data Fitting operation status */
   printStatus("dfdNewTask1D()", status);

   /* Initialize spline parameters */
   if( kind == 0 ) {
      s_order = DF_PP_LINEAR;  /* Spline is of the fourth order (cubic spline). */
      s_type = DF_PP_DEFAULT;  /* Spline is of the Bessel cubic type. */
   }
   else {
      s_order = DF_PP_CUBIC;  /* Spline is of the fourth order (cubic spline). */
      s_type = DF_PP_BESSEL;  /* Spline is of the Bessel cubic type. */
   }


   /* Define internal conditions for cubic spline construction (none in this example) */
   ic_type = DF_NO_IC;
   ic = NULL;

   /* Use not-a-knot boundary conditions. In this case, the is first and the last
   interior breakpoints are inactive, no additional values are provided. */
   bc_type = DF_BC_NOT_A_KNOT;
   bc = NULL;
   scoeffhint = DF_NO_HINT;   /* No additional information about the spline. */

   /* Set spline parameters in the Data Fitting task */
   status = dfdEditPPSpline1D( task, s_order, s_type, bc_type, bc, ic_type, ic, scoeff, scoeffhint );

   /* Check the Data Fitting operation status */
   printStatus("dfdEditPPSpline1D", status);

   /* Use a standard method to construct a cubic Bessel spline: */
   /* Pi(x) = c1,i + c2,i(x - xi) + c3,i(x - xi)2 + c4,i(x - xi)3, */
   /* The library packs spline coefficients to array scoeff: */
   /* scoeff[4*i+0] = c1,i, scoef[4*i+1] = c2,i,
    */
   /* scoeff[4*i+2] = c3,i, scoef[4*i+1] = c4,i,
    */
   /* i=0,...,N-2 */
   status = dfdConstruct1D( task, DF_PP_SPLINE, DF_METHOD_STD );

   /* Check the Data Fitting operation status */
   printStatus("dfdConstruct1D", status);


   /* Set site values (relative to x) */
   if( xi_uniform )
      xihint = DF_UNIFORM_PARTITION;     /* Partition of sites is uniform */
   else
      xihint = DF_NON_UNIFORM_PARTITION; /* Partition of sites is non-uniform */


   /* Request to compute spline values */
   ndorder = 1;
   dorder = new int[2];
   dorder[0] = 1;
   datahint = DF_NO_APRIORI_INFO;  /* No additional information about breakpoints or sites is provided. */
//   datahint = new T[4];
//   datahint[0] = 0;

   rhint = DF_MATRIX_STORAGE_ROWS; /* The library packs interpolation results in row-major format. */
   cell = NULL;                    /* Cell indices are not required. */

   /* Solve interpolation problem using the default method: compute the sline values
   at the points site(i), i=0,..., Nyi-1 and place the results to array r */
   status = dfdInterpolate1D( task, DF_INTERP, DF_METHOD_PP, Nyi, xi,
   xihint, ndorder, dorder, datahint, yi, rhint, cell );

   /* Check Data Fitting operation status */
   printStatus("dfdInterpolate1D", status);

   /* De-allocate Data Fitting task resources */
   status = dfDeleteTask( &task );

   /* Check Data Fitting operation status */
   printStatus("dfDeleteTask", status);

#ifdef DEBUG
   printf("Input partition:\n");
   for(int i=0; i<N; ++i) {
      printf("%2.2f  ", x[i]);
   }
   printf("\n");

   printf("Input data:\n");
   for(int i=0; i<N; ++i) {
      printf("%2.2f  ", y[i]);
   }
   printf("\n");

   printf("Output partition:\n");
   for(int i=0; i<2; ++i) {
      printf("%2.2f  ", xi[i]);
   }
   printf("\n");

   printf("Output data:\n");
   for(int i=0; i<Nyi; ++i) {
      printf("%2.2f,  ", yi[i]);
   }
   printf("\n");
#endif

   return 0;
}

