/**
* Part of the RealTimeCapon (RTC) library. 
*
* Author: Jon Petter Aasen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#include "BuildR.h"
#include "buildR_kernel.cuh"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <printit.h>

BuildR::BuildR(void)
{
}

BuildR::~BuildR(void)
{
}

int BuildR::getR(Complex<float>* &R_in,   // buffer holding the resulting covariance matrices
                 Complex<float>* &x_in,   // buffer holding data vectors
                 float &d,			         // diagonal loading factor
                 int &L,			         // number of spatial sublengths
                 int &Yavg,			      // number of samples averaged in time
                 int &M,			         // number of data elements
                 int &Nx,			         // number of data vectors in azimuth
                 int &Nz,			         // number of data vectors in range
                 int &Nb,			         // dimension of beamspace
                 // TODO: add suport for custom beamspace matrix
                 bool &R_on_gpu,	         // true if you want R to be left on the gpu
                 bool &x_on_gpu		      // true if x is already on the gpu
                 ) 
{

   // Check arguments
   if (Nx < 1 || Nz < 1 || L < 1 || M < 1 || Yavg < 0 || Nb < 0) {
      printIt("Error in getR: Nx, Nz, L, M, Yavg or Nb were <= 0\n");
      return cudaErrorLaunchFailure;
   }
   if (L > M) {
      printIt("Error in getR: L > M\n");
      return cudaErrorLaunchFailure;
   }
   if (2*Yavg >= Nz) {
      printIt("Error in getR: Nz <= Yavg\n");
      return cudaErrorLaunchFailure;
   }

   cuComplex *R = (cuComplex*)R_in;
   cuComplex *x = (cuComplex*)x_in;

   cuComplex* R_gpu;
   const cuComplex* x_gpu;
   cudaError e;
   size_t R_buffer_size = L*L*Nx*(Nz-2*Yavg)*sizeof(cuComplex);
   size_t x_buffer_size = M*Nx*Nz*sizeof(cuComplex);

   // Allocate memory for x and R on the gpu if needed
   if (x_on_gpu) {
      x_gpu = x;
   } else {
      cuComplex* x_gpu_tmp;
      e = cudaMalloc<cuComplex>(&x_gpu_tmp, x_buffer_size);
      if (e != cudaSuccess) return e;

      e = cudaMemcpy((void*)x_gpu_tmp, (void*)x, x_buffer_size, cudaMemcpyHostToDevice);
      if (e != cudaSuccess) return e;

      x_gpu = (const cuComplex*)x_gpu_tmp;
   }
   // TODO: If error occur, what about the allocated memory? It should be freed (if possible) before we return!

   if (R_on_gpu) {
      R_gpu = R;
   } else {
      e = cudaMalloc<cuComplex>(&R_gpu, R_buffer_size);
      if (e != cudaSuccess) return e;
   }

   // Call build R kernels
   if (Nb > 0) { // Beamspace
      // If Beamspace is performed after R is created set Nb = 0
      uint subarrayStrid = Nb;
      e = (cudaError)build_R_full(x_gpu, R_gpu, d, Nb, Yavg, Nb*(M-L+1), Nx, Nz, subarrayStrid);
   } else {
      // call kernel capable of doing sliding subarray and time averaging, and diagonal loading
      e = (cudaError)build_R(x_gpu, R_gpu, d, L, Yavg, M, Nx, Nz);
   }
   if (e != cudaSuccess) return e;

   // If needed, copy memory back to host-side and free gpu memory
   if (!R_on_gpu) {
      e = cudaMemcpy((void*)R, (void*)R_gpu, R_buffer_size, cudaMemcpyDeviceToHost);
      if (e != cudaSuccess) return e;
      e = cudaFree((void*)R_gpu);
      if (e != cudaSuccess) return e;
   }

   if (!x_on_gpu) {
      e = cudaFree((void*)x_gpu);
      if (e != cudaSuccess) return e;
   }

   return e;
}
