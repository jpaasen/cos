
#ifndef GETCAPONMEX_H
#define GETCAPONMEX_H

char usage_desc[] =
"getCaponMex()   Matlab entry point to the Cython/C++ getCapon() function.\n\
\n\
   Usage: getCaponMex(Xd, d, L, Navg, V, doFBAvg, verbose)\n\
          Xd      - XxYxM (#pixels in x and y dimension, and #channels) complex data matrix complex\n\
          d       - diagonal loading factor\n\
          L       - subarray size\n\
          Navg    - 2*Navg+1 = temporal averaging window\n\
          V       - Subspace matrix (very experimental! assume broken)\n\
          doFBAvg - true/false. Toggle forward backward averaging\n\
          verbose - true/false. Toggle verbose output";


#include <Python.h>
#include <numpy/arrayobject.h>

#include "mex.h"
#include "MexArray.h"
#include "Complex.h"

#include "stdlib.h"
#include <malloc.h>

// #ifdef __cplusplus
// extern "C" {
// #include <getCaponC.h>
// }
// #else
#include "getCaponC.h"
// #endif


// C MEX-file gateway routine (by definition):
void mexFunction(int            nlhs,   // # of..
                 mxArray       *plhs[], // left hand output arguments
                 int            nrhs,   // # of..
                 const mxArray *prhs[]) // right hand input arguments
{

#ifdef VERBOSE
   printf("Entering MEX function.");
#endif

   ComplexMexArray<double> *Xd, *V, *w, *z, *zPow; // Buffer to hold data
   RealMexArray<double> *d;
   RealMexArray<size_t> *L, *Navg;
   RealMexArray<bool> *doFBAvg, *verbose;

   // Ensure the number of inputs are correct
   if( nrhs != 7 ) {
      mexWarnMsgTxt("Number of arguments are incorrect!\n");
      mexErrMsgTxt(usage_desc);
   }

   // Import the data
   Xd      = new ComplexMexArray<double>(prhs[0]);
   d       = new RealMexArray<double>(prhs[1]);
   L       = new RealMexArray<size_t>(prhs[2]);
   Navg    = new RealMexArray<size_t>(prhs[3]);
   V       = new ComplexMexArray<double>(prhs[4]);
   doFBAvg = new RealMexArray<bool>(prhs[5]);
   verbose = new RealMexArray<bool>(prhs[6]);

   // Ensure data has correct format
   if( Xd->dims != 3 || d->size!=1 || L->size!=1 || Navg->size!=1 || doFBAvg->size!=1 || verbose->size!=1 ) {
      mexWarnMsgTxt("Data dimensions are incorrect!\n");
      mexErrMsgTxt(usage_desc);
   }
   
   size_t Nx   = Xd->shape[0];
   size_t Ny   = Xd->shape[1];
//    size_t M    = Xd->shape[2];

   size_t w_shape[3];
   w_shape[0] = Nx;
   w_shape[1] = Ny;
   w_shape[2] = L->data[0];
   
   size_t z_shape[2];
   z_shape[0] = Nx;
   z_shape[1] = Ny;
   
   size_t V_shape[2];
   V_shape[0] = 1;
   V_shape[1] = 1;

#ifdef DEBUG
   printf("Initializing Python interpreter");
#endif
   // To use Python, we first need to start the Python interpreter
   Py_Initialize();

#ifdef DEBUG
   printf("Creating Pyarrays");
#endif

   // Before doing anything numpy-related, we need to call import_array()
   import_array();

   // Wrap our data into a numpy object
   PyArrayObject *Xd_py = (PyArrayObject*)PyArray_SimpleNewFromData(Xd->dims, (npy_intp*)Xd->shape, NPY_COMPLEX128, (void*)Xd->data);
   PyArrayObject *V_py  = (PyArrayObject*)PyArray_SimpleNewFromData(V->dims,  (npy_intp*)V_shape,   NPY_COMPLEX128, (void*)V->data);

   // ...and create some empty numpy objects for the results
   PyArrayObject *w_py    = (PyArrayObject*)PyArray_SimpleNew(3, (npy_intp*)w_shape, NPY_COMPLEX128);
   PyArrayObject *z_py    = (PyArrayObject*)PyArray_SimpleNew(2, (npy_intp*)z_shape, NPY_COMPLEX128);
   PyArrayObject *zPow_py = (PyArrayObject*)PyArray_SimpleNew(2, (npy_intp*)z_shape, NPY_COMPLEX128);

   // For each Python object we create the Python reference counter must be incremented, to tell Python
   // which objects it need to "free" when we call Py_Finalize() (later on)
   Py_INCREF(Xd_py);
   Py_INCREF(V_py);
   Py_INCREF(w_py);
   Py_INCREF(z_py);
   Py_INCREF(zPow_py);


#ifdef VERBOSE
   printf("Running getCapon");
#endif

   // Call the getCapon() method defined in getCaponC.h, which is created by Cython when the function is defined 'public'
   // We'll call the supplied init-method as well.
//#ifdef USE_MKL
//   initgetCaponC_MKL();
//#else
//   initgetCaponC();
//#endif

   initgetCaponC();
   getCapon(z_py, w_py, zPow_py, Xd_py, d->data[0], L->data[0], Navg->data[0], V_py, false, false);

   // Create the output variables
   w    = new ComplexMexArray<double>(3, w_shape, Xd->mx_class);
   z    = new ComplexMexArray<double>(2, z_shape, Xd->mx_class);
   zPow = new ComplexMexArray<double>(2, z_shape, Xd->mx_class);

   // Replace the data with that of the Python objects (a bit ugly, should be improved)
   w->data    = (Complex<double>*)PyArray_DATA(w_py);
   z->data    = (Complex<double>*)PyArray_DATA(z_py);
   zPow->data = (Complex<double>*)PyArray_DATA(zPow_py);

   // Assign output variables to Matlab
   nlhs = 3;
   plhs[0] = z->copy();
   plhs[1] = zPow->copy();
   plhs[2] = w->copy();

   // Should this be done? Or does Python do it?
//   delete Xd_py, delete V_py, delete w_py, delete z_py, delete zPow_py;

   // Let Python clean up after the party.
   if( Py_IsInitialized() ){
      Py_Finalize();
   }

   // Cleanup
   delete Xd, delete d, delete L, delete Navg, delete V, delete doFBAvg, delete verbose, delete w, delete z, delete zPow;

}


#endif
