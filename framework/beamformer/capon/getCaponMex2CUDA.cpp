
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


#include "mex.h"
#include "MexArray.h"
#include "Complex.h"

#include "stdlib.h"
#include <malloc.h>

#include "Capon.h"
// C MEX-file gateway routine (by definition):
void mexFunction(int            nlhs,   // # of..
                 mxArray       *plhs[], // left hand output arguments
                 int            nrhs,   // # of..
                 const mxArray *prhs[]) // right hand input arguments
{

#ifdef VERBOSE
   printf("Entering MEX function.");
#endif

   ComplexMexArray<float> *Xd, *V, *w, *z, *R; // Buffer to hold data
   RealMexArray<float> *d;
   RealMexArray<int> *L, *Navg;
   RealMexArray<bool> *doFBAvg, *verbose;

   // Ensure the number of inputs are correct
   if( nrhs != 7 ) {
      mexWarnMsgTxt("Number of arguments are incorrect!\n");
      mexErrMsgTxt(usage_desc);
   }

   // Import the data
   Xd      = new ComplexMexArray<float>(prhs[0]);
   d       = new RealMexArray<float>(prhs[1]);
   L       = new RealMexArray<int>(prhs[2]);
   Navg    = new RealMexArray<int>(prhs[3]);
   V       = new ComplexMexArray<float>(prhs[4]);
   doFBAvg = new RealMexArray<bool>(prhs[5]);
   verbose = new RealMexArray<bool>(prhs[6]);

   // Ensure data has correct format
   if( Xd->dims != 3 || d->size!=1 || L->size!=1 || Navg->size!=1 || doFBAvg->size!=1 || verbose->size!=1 ) {
      mexWarnMsgTxt("Data dimensions are incorrect!\n");
      mexErrMsgTxt(usage_desc);
   }
   
   int Nx   = Xd->shape[0];
   int Ny   = Xd->shape[1];
   int M    = Xd->shape[2];

   int newV = 0;

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

   size_t R_shape[4];
   R_shape[0] = Nx;
   R_shape[1] = Ny;
   R_shape[2] = L->data[0];
   R_shape[3] = L->data[0];


//   R_py = np.zeros((Nx,Nz-2*Yavg,L,L),dtype=np.complex64)
//   z_py = np.zeros((Nx,Nz-2*Yavg,   ),dtype=np.complex64)
//   w_py = np.zeros((Nx,Nz-2*Yavg,L  ),dtype=np.complex64)


   // Create the output variables
   w    = new ComplexMexArray<float>(3, w_shape, Xd->mx_class);
   z    = new ComplexMexArray<float>(2, z_shape, Xd->mx_class);
   R    = new ComplexMexArray<float>(4, R_shape, Xd->mx_class);

   Capon capon;

   capon.getCapon(
         z->data,   // output amplitude per pixel
         w->data,   // output weights per pixel
         R->data,   // buffer holding the resulting covariance matrices
         Xd->data,   // buffer holding data vectors
         d->data[0],                // diagonal loading factor
         L->data[0],                  // number of spatial sublengths
         Navg->data[0],               // number of samples averaged in time
         M,                  // number of data elements
         Nx,                 // number of data vectors in azimuth
         Ny,                 // number of data vectors in range
         newV                  // dimension of beamspace
         // TODO: add suport for custom beamspace matrix
        );


   // Assign output variables to Matlab
   nlhs = 3;
   plhs[0] = z->copy();
   plhs[1] = R->copy();
   plhs[2] = w->copy();

   // Cleanup
   delete Xd, delete d, delete L, delete Navg, delete V, delete doFBAvg, delete verbose, delete w, delete z, delete R;

}


#endif
