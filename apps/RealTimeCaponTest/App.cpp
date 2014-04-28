#include "App.h"

// includes from libRealTimeCapon
//#include <BuildR.h>
//#include <Solver.h>
#include <Capon.h>
#include <CudaUtils.h>

#include "TestData.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuComplex.h>
#include <Complex.h>
#include <ctime>

#include "sdkHelper.h"

App::App(void)
{
}

App::~App(void)
{
}

int main()
{
   // Select device (mostly for debuging on non-desplay device)
   // select device
   int dev = 0;
   cudaGetDeviceCount(&dev);
   cudaError e = cudaSetDevice(dev-1);

	// Test framework objects
	//BuildR builder;
	//Solver solver;
	Capon capon;

   //*** TESTING BUILD R ***//

   int M    = M32;//M16;                           // data vector length from TestData.h
   int L    = 46;//32;//24;//M/2;                           // sub array length
   int Yavg = 0;                             // time avg. Not in use by the current implementation
   int Nx   = 120;//100;//6;//976;//10;//70;//1;//5;              // number of samples in x simulating a dataset of 70 tx and 832 range samples (csound data)
   int Ny   = 540;//832;//10;//154;//3;//832;//2;//1;          // number of samples in z
   float d  = 0.01f; //1.0f;                 // diagonal loading factor
   int Nb   = 0;//3;//3;//32;//17;//8;//0;                             // size of beamspace (0 == full space)

   int N = (Ny-2*Yavg)*Nx * 3;//2; // total number of samples (factor 2 is added to get enough data for M = 64)
   int Ndata = Ny*Nx * 3;//2;

   Complex<float>* testX = (Complex<float>*)testX32;  // testdata from TestData.h
   //Complex<float>* testX = (Complex<float>*)testX16;  // testdata from TestData.h
	//Complex<float>* testX = (Complex<float>*)testImag;  // testdata from TestData.h

   Complex<float>* R = new Complex<float>[N*L*L];
	Complex<float>* x = new Complex<float>[Ndata*M];

   //#pragma omp parallel for
	for (int i = 0; i < M*Ndata; ++i) { // fill x with Nx*Ny copies
      if (M == 16 && i < M) {
         x[i] = Complex<float>(testRightside[i].x, testRightside[i].y);
		} else if (M == 16 && i/M == 2) { 
			x[i] = Complex<float>(testImag[i%M].x, testImag[i%M].y); // TODO: There is an issue when using imag x with buildR
		} else if (M == 32 && i/M == 3) {
         x[i] = testX[0];
      } else {
		   x[i] = testX[i%M];
      }
   }

   /***/
    N = (Ny-2*Yavg)*Nx;//61000; // testing setup for csound data (M = 64, L = 16)
    M = 96;//64;
    L = 46;//32;//32;//16;
    Nb = 0;//1;
    delete[] R;
    R = new Complex<float>[N*L*L];
   /***/
   Complex<float>* z = new Complex<float>[N];
   Complex<float>* w = new Complex<float>[N*L];
   // TODO: Allocate memory for P and w. Remember to free them after use.

   cuUtilsSafeCall( (cudaError)capon.getCapon(z, w, R, x, d, L, Yavg, M, Nx, Ny, Nb) , true);

	// The colums of R is layed out in memory using strid N.
	// print R's
	//int n_print = 0;
   //int n_print = 10;
	int n_print = N;
	for (int i = 0; i < n_print; ++i) {
		for (int j = 0; j < L; ++j) {
			for (int k = 0; k < L; ++k) {

            //printf("(%4.1f, %4.1f)", R[j*L*N + i*L + k].re, R[j*L*N + i*L + k].im); // row sride of N
            
				if (i == 0)
					printf("(%4.1f, %4.1f)", R[j*L + i*L*L + k].re, R[j*L + i*L*L + k].im); // row stride of 1
				
				Complex<float> firstRElem = R[j*L + k];
				Complex<float> currentRElem = R[j*L + i*L*L + k];

				if (firstRElem.re != currentRElem.re || firstRElem.im != currentRElem.im) {
					printf("Error in R(%d,%d,%d), was (%f,%f) expected (%f,%f)\n", i, j, k, currentRElem.re, currentRElem.im, firstRElem.re, firstRElem.im);
				}
         
         }
			if (i == 0)
				printf("\n");
		}
		if (i == 0)
			printf("\n");
	}

   // Print w's
   for (int i = 0; i < 10/*N*/; ++i) {
      for (int j = 0; j < L; ++j) {
         printf("(%1.4f, %1.4f) ", w[i*L + j].re, w[i*L + j].im);
      }
      printf("\n");
   }

   // Print z's
   for (int i = 0; i < 10/*N*/; ++i) {
      printf("(%1.4f, %1.4f)\n", z[i].re, z[i].im);
   }

   delete[] x;
	delete[] R;
   delete[] z;
   delete[] w;
}
