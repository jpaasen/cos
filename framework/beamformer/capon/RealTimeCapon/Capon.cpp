/**
* Part of the RealTimeCapon (RTC) library. 
*
* Author: Jon Petter Aasen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#define WRITE_TIMING_TO_FILE true   // Write timing of each step to file.
//#define SLIDING_BEAMSPACE true      // select to compile with sliding beamspace or unselect to compile with beamspace processing using matrix multiplication.
#define COPY_R_DEVICE_HOST true     // R is copied from device to host. This should be turned off if speed is essential.

#include "Capon.h"

#include "BuildR.h"
#include "Solver.h"
#include "CudaUtils.h"
#include "capon_kernel.cuh"
#include "sliding_beamspace_kernel.cuh"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include "sdkHelper.h" // Used for timing

#include <stdexcept>

//#ifndef SLIDING_BEAMSPACE
#include <cublas_v2.h>
//#endif
#include <cuda_profiler_api.h>

typedef unsigned int uint;

void startTimer(StopWatchInterface* &timer) 
{
   sdkCreateTimer( &timer );
   sdkStartTimer( &timer );
}

float stopAndGetTimeMs(StopWatchInterface* &timer) 
{
   sdkStopTimer( &timer );
   float elapsedTimeInMs = sdkGetTimerValue( &timer );
   sdkDeleteTimer( &timer );
   timer = NULL;
   return  elapsedTimeInMs;
}

Capon::Capon(void)
{
   // for testing
   //cudaError e = cudaSetDevice(0);

   x_gpu   = NULL;  // data vectors
   R_gpu   = NULL;  // covariance matrices
   Ria_gpu = NULL;  // solutions
   a_gpu   = NULL;  // right hand sides
   z_gpu   = NULL;  // amplitude
   w_gpu   = NULL;  // weights

   cur_N    = init_N;     // current number of samples (N) supported by buffer
   cur_Npx  = init_Npx;   // current number of pixels (Npx) supported by buffer
   cur_M    = init_M;     // current number of data sample elements (M) supported by buffer
   cur_L    = init_L;     // current number of subarray elements (L) supported by buffers
}

Capon::~Capon(void)
{
   freeBuffers();
}

int Capon::getCapon(
      Complex<float>* &z,   // output amplitude per pixel
      Complex<float>* &w,   // output weights per pixel
      Complex<float>* &R,   // buffer holding the resulting covariance matrices
      Complex<float>* &x,   // buffer holding data vectors
      float &d,                // diagonal loading factor
      int &L,                  // number of spatial sublengths
      int &Yavg,               // number of samples averaged in time
      int &M,                  // number of data elements
      int &Nx,                 // number of data vectors in azimuth
      int &Nz,                 // number of data vectors in range
      int &Nb                  // dimension of beamspace
      // TODO: add suport for custom beamspace matrix
     )
{
//#ifdef PROFILE
//   cudaProfilerStart();
//#endif

   cudaError e = cudaSuccess;
   int N    = Nx * Nz;           // total number of samples
   int Npx  = Nx * (Nz-2*Yavg);  // total number of pixels

   printIt("N = %d, M = %d, L = %d, K = %d, Nb = %d\n", N, M, L, Yavg, Nb);

   // Check arguments
   if (Nx < 1 || Nz < 1 || L < 1 || M < 1 || Yavg < 0 || Nb < 0) {
      printIt("Error in Capon GPU: Nx or Nz were < 1, or L or M < 1, or Yavg or Nb < 0\n");
      return cudaErrorLaunchFailure;
   }

   if (Nb > L) {
      printIt("Error in Capon GPU: Nb can not be larger than L (L=%d, Nb=%d)\n", L, Nb);
      return cudaErrorLaunchFailure;
   }

   // Check if we need larger gpu buffers to handle the problem
   if (N > cur_N || Npx > cur_Npx || M > cur_M || L > cur_L) {

      printIt("N, M or L is larger than init values. Allocating larger buffers\n");
      size_t buffer_size = initBufferSize(N, Npx, M, L);
      freeBuffers();
      
      if (largerThanMaxSupportedSize(buffer_size)) { // not enough memory available on selected device

         // TODO: auto-loop kernelsls 
         printIt("Not enought device memory for selected N, M and L. Try to divide your problem into smaller segments. Reseting object to inital values.\n");
         initBufferSize(init_N, init_Npx, init_M, init_L);
         checkAndInitGpuBuffers();
         return cudaErrorMemoryValueTooLarge;

      } else { // allocate memory and run

         e = (cudaError) checkAndInitGpuBuffers();

         if (e != cudaSuccess) {
            printIt("Error in Capon GPU: Error allocating device memory. Reseting object to inital values.\n");
            freeBuffers();
            initBufferSize(init_N, init_Npx, init_M, init_L);
            checkAndInitGpuBuffers();
            return cudaErrorMemoryValueTooLarge;
         } else {
            e = (cudaError) getCaponSegment(z, w, R, x, d, L, Yavg, M, Nx, Nz, Nb);
            if (e != cudaSuccess) {
               printIt("Error calling getCaponSegment after reallocating memory: %s\n", cudaGetErrorString(e));
            }
            return e;
         }
      }
   } else { // problem size is ok with respect to memory, just run it
      e = (cudaError) getCaponSegment(z, w, R, x, d, L, Yavg, M, Nx, Nz, Nb);
      if (e != cudaSuccess) {
         printIt("Error calling getCaponSegment: %s\n", cudaGetErrorString(e));
      }
      return e;
   }
//#ifdef PROFILE
//   cudaProfilerStop();
//#endif
}

// TODO: Wrap this function by a new function that calls this one in a loop if N is too large
int Capon::getCaponSegment(
      Complex<float>* &z,   // output amplitude per pixel
      Complex<float>* &w,   // output weights per pixel
      Complex<float>* &R,   // buffer holding the resulting covariance matrices
      Complex<float>* &x,   // buffer holding data vectors
      float &d,                // diagonal loading factor
      int &L,                  // number of spatial sublengths
      int &Yavg,               // number of samples averaged in time
      int &M,                  // number of data elements
      int &Nx,                 // number of data vectors in azimuth
      int &Nz,                 // number of data vectors in range
      int &Nb                  // dimension of beamspace
      // TODO: add suport for custom beamspace matrix
     )
{
   cudaError e;
   StopWatchInterface *timer = NULL; // timer object
   float elapsedTimeInMs = 0.0f;

#ifdef WRITE_TIMING_TO_FILE
   int timer_idx = 0;
   float timer_values[10];
   for(int i=0; i<10; ++i) {
      timer_values[i] = 0.0;
   }
#endif

   // Framework objects
   BuildR builder;
   Solver solver;

   int N   = Nx * Nz;            // total number of samples
   int Npx = Nx * (Nz-2*Yavg);   // total number of pixels

   // Copy x from host to device
   //size_t R_buffer_size = L*L*Npx*sizeof(Complex<float>);
   //size_t x_buffer_size = M*N*sizeof(Complex<float>);
   bool x_on_gpu = true;
   bool R_on_gpu = true;
   ::startTimer(timer);
   e = cudaMemcpy((void*)x_gpu, (void*)x, x_buffer_size, cudaMemcpyHostToDevice);
   if (e != cudaSuccess) {return e;}
   elapsedTimeInMs = ::stopAndGetTimeMs(timer);
#ifdef WRITE_TIMING_TO_FILE
   timer_values[timer_idx++] = elapsedTimeInMs; // timing memcpy data (x) cpu->gpu
#endif

#ifdef SLIDING_BEAMSPACE
   Complex<float> *s_bs, *x_bs, *x_gpu_tmp;
   if (Nb > 0) {
      // Alloc memory for sliding beamspace transformation
      size_t x_bs_size = N*Nb*(M-L+1)*sizeof(Complex<float>);
      e = cudaMalloc<Complex<float> >(&x_bs, x_bs_size); // TODO: This memory is in addition to inital size!!! No good!
      if (e != cudaSuccess) {return e;}
      e = cudaMalloc<Complex<float> >(&s_bs, N*Nb*sizeof(Complex<float>)); // TODO: This memory is in addition to inital size!!! No good! 
      if (e != cudaSuccess) {return e;}

      // Sliding beamspace transformation
      ::startTimer(timer);
      e = (cudaError) sliding_beamspace((cuComplex*)x_gpu, (cuComplex*)x_bs, (cuComplex*)s_bs, M, L, Nb, Nx, Nz);
      if (e != cudaSuccess) {return e;}
      cudaThreadSynchronize();
      elapsedTimeInMs = ::stopAndGetTimeMs(timer);
      printIt("Time used for sliding beamspace transformation: %f ms\n", elapsedTimeInMs);

      e = cudaGetLastError();
      if (e != cudaSuccess) {return e;}

      // Move result to x_gpu buffer to intergrate with call to buildR object
      x_gpu_tmp = x_gpu;
      x_gpu = x_bs;
   }
#endif
#ifdef WRITE_TIMING_TO_FILE
#ifdef SLIDING_BEAMSPACE
   timer_values[timer_idx++] = elapsedTimeInMs; // timing sliding beamspace transform
#else
   timer_values[timer_idx++] = 0.0;
#endif
#endif

   // Build R's
   ::startTimer(timer);
#ifdef SLIDING_BEAMSPACE
   e = (cudaError) builder.getR(R_gpu, x_gpu, d, L, Yavg, M, Nx, Nz, Nb, R_on_gpu, x_on_gpu);
#else
   int zero = 0;
   e = (cudaError) builder.getR(R_gpu, x_gpu, d, L, Yavg, M, Nx, Nz, zero, R_on_gpu, x_on_gpu);
#endif
   if (e != cudaSuccess) {return e;}
   e = cudaDeviceSynchronize();
   if (e != cudaSuccess) {return e;}
   elapsedTimeInMs = ::stopAndGetTimeMs(timer);
#ifdef WRITE_TIMING_TO_FILE
   timer_values[timer_idx++] = elapsedTimeInMs; // timing buildR
#endif
   printIt("Time to build all R: %f ms\n", elapsedTimeInMs);

   // Copy R from device to host
   ::startTimer(timer);
#ifndef SLIDING_BEAMSPACE
#ifdef COPY_R_DEVICE_HOST
   e = cudaMemcpy((void*)R, (void*)R_gpu, R_buffer_size, cudaMemcpyDeviceToHost);
   if (e != cudaSuccess) {return e;}
#endif
#endif
   elapsedTimeInMs = ::stopAndGetTimeMs(timer);
#ifdef WRITE_TIMING_TO_FILE
   timer_values[timer_idx++] = elapsedTimeInMs; // timing memcpy R gpu->cpu
#endif

   // Initialize steering vector 'a_gpu' with ones
   ::startTimer(timer);
#ifdef SLIDING_BEAMSPACE
   if (Nb > 0) {
      e = (cudaError) values((cuComplex*)a_gpu, 1.0f, (Npx*Nb), Nb, true);
   } else {
      e = (cudaError) values((cuComplex*)a_gpu, 1.0f, (Npx*L), 0, false);
   }
#else
   e = (cudaError) values((cuComplex*)a_gpu, 1.0f, (uint)(Npx*L), 0, false);
#endif
   if (e != cudaSuccess) {return e;}
   e = cudaDeviceSynchronize();
   if (e != cudaSuccess) {return e;}
   elapsedTimeInMs = ::stopAndGetTimeMs(timer);
#ifdef WRITE_TIMING_TO_FILE
   timer_values[timer_idx++] = elapsedTimeInMs; // timing init a on gpu
#endif

   // using CUBLAS for beamspace processing // TODO: Refactor to own function 
   uint tmpL;
   Complex<float> *B;
   if (Nb > 0) {
      // init butler matrix
      e = cudaMalloc<Complex<float> >(&B, 1*Nb*L*sizeof(Complex<float>)); // TODO: This memory is in addition to inital size!!! No good! Even if it is now just one value.
      if (e != cudaSuccess) {return e;}
      e = (cudaError) butler_matrix((cuComplex*)B, Nb, L, 1);
      if (e != cudaSuccess) {return e;}
   }
#ifndef SLIDING_BEAMSPACE
   Complex<float> *Ba, *tmpR, *tmpBRBH, *tmpRBH, *tmpRia, *tmpa; 
   if (Nb > 0) { // Beamspace, init extra memory

      // init memory for tmpR = R*B^H
      e = cudaMalloc<Complex<float> >(&tmpRBH, Npx*L*Nb*sizeof(Complex<float>)); // TODO: This memory is in addition to inital size!!! No good!
      if (e != cudaSuccess) {return e;}

      // init memory for B*tmpR
      e = cudaMalloc<Complex<float> >(&tmpBRBH, Npx*Nb*Nb*sizeof(Complex<float>)); // TODO: This memory is in addition to inital size!!! No good!
      if (e != cudaSuccess) {return e;}
   } 
#endif

   ::startTimer(timer);
#ifndef SLIDING_BEAMSPACE
   if (Nb > 0) { // beamspace, do transformation
      // calc tmpR = R*B^H
      e = (cudaError)batchMultiplyMatrices(R_gpu, CUBLAS_OP_N, B, CUBLAS_OP_C, tmpRBH, L, L, Nb, Npx, L, Nb, L, false, true);
      if (e != cudaSuccess) {return e;}

      // calc B*tmpR
      e = (cudaError)batchMultiplyMatrices(B, CUBLAS_OP_N, tmpRBH, CUBLAS_OP_N, tmpBRBH, Nb, L, Nb, Npx, Nb, L, Nb, true, false);
      if (e != cudaSuccess) {return e;}
   }
#endif
   elapsedTimeInMs = ::stopAndGetTimeMs(timer);
   if (Nb > 0) { // beamspace, prepare for solving
#ifndef SLIDING_BEAMSPACE
      e = cudaFree(tmpRBH);
      if (e != cudaSuccess) {return e;}
      tmpR = R_gpu; // save original R_gpu-buffer for later use
      R_gpu = tmpBRBH;

      // calc a_bs = B*a = sqrt(L) * [1 0 ... 0]
      e = cudaMalloc<Complex<float> >(&Ba, Npx*Nb*sizeof(Complex<float>)); // TODO: This memory is in addition to inital size!!! No good!
      if (e != cudaSuccess) {return e;}
      e = (cudaError)batchMultiplyMatrices(B, CUBLAS_OP_N, a_gpu, CUBLAS_OP_N, Ba, Nb, L, 1, Npx, Nb, L, Nb, true, false);
      if (e != cudaSuccess) {return e;}
    
      tmpa = a_gpu; // save original a-buffer for later use
      a_gpu = Ba;

      tmpRia = Ria_gpu;
      e = cudaMalloc<Complex<float> >(&Ria_gpu, Npx*Nb*sizeof(Complex<float>)); // TODO: This memory is in addition to inital size!!! No good!
      if (e != cudaSuccess) {return e;}
#endif
      tmpL = L;
      L = Nb; // Change system dimensions
   }
#ifdef WRITE_TIMING_TO_FILE
   timer_values[timer_idx++] = elapsedTimeInMs; // timing beamspace part 1
#endif
   printIt("Time for beamspace part 1: %f ms\n", elapsedTimeInMs);

   // Find Ria = R^(-1)a by solving Rb = a for b:
   bool Ria_on_gpu = true;
   bool a_on_gpu = true;
   ::startTimer(timer);
   e = (cudaError) solver.solve(Ria_gpu, R_gpu, a_gpu, L, Npx, Ria_on_gpu, R_on_gpu, a_on_gpu);
   if (e != cudaSuccess) {return e;}
   e = cudaDeviceSynchronize();
   if (e != cudaSuccess) {return e;}
   elapsedTimeInMs = ::stopAndGetTimeMs(timer);
#ifdef WRITE_TIMING_TO_FILE
   timer_values[timer_idx++] = elapsedTimeInMs; // timing solver
#endif
   printIt("Time to solve all R\\b: %f ms\n", elapsedTimeInMs);

   ::startTimer(timer);
#ifndef SLIDING_BEAMSPACE
   if (Nb > 0) { // Beamspace, transfer weights back to element space (Ria = B^H Ria_BS)
      L = tmpL; // change L back to original value
      e = (cudaError)batchMultiplyMatrices(B, CUBLAS_OP_C, Ria_gpu, CUBLAS_OP_N, tmpRia, L, Nb, 1, Npx, Nb, Nb, L, true, false);
      if (e != cudaSuccess) {return e;}
   }
#endif
   elapsedTimeInMs = ::stopAndGetTimeMs(timer); 
   if (Nb > 0) { // Beamspace, release gpu memory
#ifndef SLIDING_BEAMSPACE
      e = cudaFree(Ria_gpu);
      if (e != cudaSuccess) {return e;}
      Ria_gpu = tmpRia;

      e = cudaFree(R_gpu);
      if (e != cudaSuccess) {return e;}
      R_gpu = tmpR;

      e = cudaFree(a_gpu);
      if (e != cudaSuccess) {return e;}
      a_gpu = tmpa;
#endif
   }
#ifdef WRITE_TIMING_TO_FILE
   timer_values[timer_idx++] = elapsedTimeInMs; // timing beamspace part 2
#endif
   printIt("Time for beamspace part 2: %f ms\n", elapsedTimeInMs);

   // Run the kernel in charge of the computing weights and beamformer output
   ::startTimer(timer);
   if (Nb == 0) {
      getCaponOutput((cuComplex*)z_gpu, (cuComplex*)w_gpu, (cuComplex*)x_gpu, (cuComplex*)Ria_gpu, M, L, Yavg, Nz, Npx);
   } else {
#ifdef SLIDING_BEAMSPACE
      getCaponOutputFromSubarraySums((cuComplex*)z_gpu, (cuComplex*)Ria_gpu, (cuComplex*)s_bs, (cuComplex*)Ria_gpu, tmpL, L, Yavg, Nz, Npx, true);
#else
      getCaponOutput((cuComplex*)z_gpu, (cuComplex*)w_gpu, (cuComplex*)x_gpu, (cuComplex*)Ria_gpu, M, L, Yavg, Nz, Npx);
#endif
   }
   e = cudaDeviceSynchronize();
   if (e != cudaSuccess) {return e;}
   elapsedTimeInMs = ::stopAndGetTimeMs(timer);
#ifdef WRITE_TIMING_TO_FILE
   timer_values[timer_idx++] = elapsedTimeInMs; // timing norm weights and calc beamformer ouput
#endif
   printIt("Time to calc amplitude for all pixels: %f ms\n", elapsedTimeInMs);

#ifdef SLIDING_BEAMSPACE
   if (Nb > 0) { // Beamspace, transfer weights back to element space (Ria = B^H Ria_BS)
      L = tmpL; // change L back to original value
      e = (cudaError)batchMultiplyMatrices(B, CUBLAS_OP_C, Ria_gpu, CUBLAS_OP_N, w_gpu, L, Nb, 1, Npx, Nb, Nb, L, true, false);
      if (e != cudaSuccess) {return e;}
   }
#endif

   if (Nb > 0) {
      e = cudaFree(B);
      if (e != cudaSuccess) {return e;}
   }

   // Copy z_gpu / w_gpu from device to host
   ::startTimer(timer);
   e = cudaMemcpy((void*)z, (void*)z_gpu, z_buffer_size, cudaMemcpyDeviceToHost);
   if (e != cudaSuccess) {return e;}
   e = cudaMemcpy((void*)w, (void*)w_gpu, w_buffer_size, cudaMemcpyDeviceToHost);
   if (e != cudaSuccess) {return e;}
   elapsedTimeInMs = ::stopAndGetTimeMs(timer);
#ifdef WRITE_TIMING_TO_FILE
   timer_values[timer_idx++] = elapsedTimeInMs; // timing copy weights and output gpu->cpu
#endif

#ifdef SLIDING_BEAMSPACE
   if (Nb > 0) {
      e = cudaFree(s_bs);
      if (e != cudaSuccess) {return e;}
      e = cudaFree(x_gpu);
      if (e != cudaSuccess) {return e;}
      x_gpu = x_gpu_tmp;
   }
#endif

#ifdef WRITE_TIMING_TO_FILE
   // write running times to file
   FILE *pFile = fopen("timer_values.txt", "a");   
   for (int i=0 ; i<timer_idx; i++) {
     fprintf (pFile, "%f ",timer_values[i] );
   }
   fprintf (pFile, "\n" );
   fclose (pFile);

   // write running time info to file
   pFile = fopen("timer_values_info.txt", "a");
   fprintf(pFile, 
      "| Memcpy x H->D | Sliding beamspace | Build R | Memcpy R D->H | Init a | matrix beamspace part 1 | Solve R^-1b=a | matrix beamspace part 2 | Calc w,z | Memcpy w,z D->H |\n");
   fclose (pFile);
#endif

   return e;
}

int Capon::checkAndInitGpuBuffers() 
{
   // N should always be > Npx since Npx = Nx*(Nz - 2*Yavg) and N = Nx*Nz.
   // It is therefore ok to allocate buffers based on N, but we will use more memory than we actually need.

   cudaError e = cudaSuccess;

   if (largerThanMaxSupportedSize(this->total_buffersize)) {
      return cudaErrorMemoryValueTooLarge;
   }

   if (this->R_gpu == NULL) {
      e = cudaMalloc<Complex<float> >(&R_gpu, this->R_buffer_size);
      if (e != cudaSuccess) {return e;}
   }

   if (this->x_gpu == NULL) {
      e = cudaMalloc<Complex<float> >(&x_gpu, this->x_buffer_size);
      /*cuComplex* tmp_x = (cuComplex*)x_gpu;
      e = cudaMalloc<cuComplex>(&tmp_x, this->x_buffer_size);
      x_gpu = (Complex<float>*) tmp_x;*/
      if (e != cudaSuccess) {return e;}
   }

   if (this->Ria_gpu == NULL) {
      e = cudaMalloc<Complex<float> >(&Ria_gpu, this->Ria_buffer_size);
      if (e != cudaSuccess) {return e;}
   }
   
   if (this->a_gpu == NULL) {
      e = cudaMalloc<Complex<float> >(&a_gpu, this->a_buffer_size);
      if (e != cudaSuccess) {return e;}
   }

   if (this->w_gpu == NULL) {
      e = cudaMalloc<Complex<float> >(&w_gpu, this->w_buffer_size);
      if (e != cudaSuccess) {return e;}
   }

   if (this->z_gpu == NULL) {
      e = cudaMalloc<Complex<float> >(&z_gpu, this->z_buffer_size);
      if (e != cudaSuccess) {return e;}
   }
   return e;
}

void Capon::freeBuffers()
{
   if (a_gpu != NULL)   cudaFree(a_gpu);
   if (R_gpu != NULL)   cudaFree(R_gpu);
   if (z_gpu != NULL)   cudaFree(z_gpu);
   if (w_gpu != NULL)   cudaFree(w_gpu);
   if (x_gpu != NULL)   cudaFree(x_gpu);
   if (Ria_gpu != NULL) cudaFree(Ria_gpu);

   a_gpu = NULL;
   R_gpu = NULL;
   z_gpu = NULL;
   w_gpu = NULL;
   x_gpu = NULL;
   Ria_gpu = NULL;
}

size_t Capon::initBufferSize(int N, int Npx, int M, int L)
{
   x_buffer_size   = M*N     * sizeof(cuComplex);//sizeof(Complex<float>);//
   // If temporal averaging is applied all buffers below is too large ().
   R_buffer_size   = L*L*Npx * sizeof(Complex<float>);
   Ria_buffer_size = L*Npx   * sizeof(Complex<float>);
   a_buffer_size   = L*Npx   * sizeof(Complex<float>);
   w_buffer_size   = L*Npx   * sizeof(Complex<float>);
   z_buffer_size   = Npx     * sizeof(Complex<float>);

   this->total_buffersize = R_buffer_size + x_buffer_size + 
      Ria_buffer_size + a_buffer_size +
      w_buffer_size + z_buffer_size;

   this->cur_N = N;
   this->cur_M = M;
   this->cur_L = L;

   return this->total_buffersize;
}

bool Capon::largerThanMaxSupportedSize(size_t total_memory)
{
   if (total_memory == 0) {return true;}

   size_t free, total;
   cudaError e = cudaMemGetInfo(&free, &total);
   if (e != cudaSuccess) {
      printIt("Error in cudaMemGetInfo: %s\n", cudaGetErrorString(e));
      return true;
   }

   printIt("Free: %d MiB (%d MB), Total: %d MiB (%d MB)\n", ((int)free)/1048576, free/1000000, total/1048576, total/1000000);
   printIt("Available memory: %d MB, Required memory: %d MB\n", free/1000000, total_memory/1000000);

   if (free < total_memory) {
      return true;
   }
   return false;
}

/** 
* Helper function that converts matrices in linear memory to pointer array and transfers it to the GPU. 
* Remember to free GPU memory returned by this function using cudaFree().
* 
* oneToOne: 
*  if true  A contains N matrices and return value is N pointers to these matrices. 
*  if false A contains 1 matrix and return value is N pointers to this matrix.
**/
cuComplex** getPointerArray(cuComplex* A, int Nrow, int Ncol, int N, bool oneToOne) 
{
   cuComplex** A_ptr = new cuComplex*[N];
   for (int i = 0; i < N; ++i) {
      if (oneToOne) {
         A_ptr[i] = A + (Nrow*Ncol*i);
      } else {
         A_ptr[i] = A;
      }
   }

   size_t N_bytes = N*sizeof(cuComplex*);
   
   cuComplex** A_ptr_gpu;
   cudaMalloc<cuComplex*>(&A_ptr_gpu, N_bytes);
   cudaMemcpy(A_ptr_gpu, A_ptr, N_bytes, cudaMemcpyHostToDevice);
   
   delete[] A_ptr;

   return A_ptr_gpu;
}

int Capon::batchMultiplyMatrices(Complex<float>* A, int opA, 
                                 Complex<float>* B, int opB, 
                                 Complex<float>* C, 
                                 int M, int K, int N, int Npx, 
                                 int lda, int ldb, int ldc,
                                 bool duplicateA, bool duplicateB)
{
   const cuComplex** A_ptr = (const cuComplex**)getPointerArray((cuComplex*)A, M, K, Npx, !duplicateA);
   const cuComplex** B_ptr = (const cuComplex**)getPointerArray((cuComplex*)B, K, N, Npx, !duplicateB);
   cuComplex** C_ptr = getPointerArray((cuComplex*)C, M, N, Npx, true);
   
   cuComplex alpha = make_cuComplex(1.0f, 0.0f);
   cuComplex beta  = make_cuComplex(0.0f, 0.0f);

   cublasStatus_t status;
   cublasHandle_t handle;

   status = cublasCreate(&handle);
   if (status != CUBLAS_STATUS_SUCCESS) {
      printIt("Error in CUBAS handle init.\n");
      return status;
   } 

   status = cublasCgemmBatched(
      handle, (cublasOperation_t)opA, (cublasOperation_t)opB, 
      M, N, K, &alpha, A_ptr, lda, B_ptr, ldb, &beta, C_ptr, ldc, Npx);

   if (status != CUBLAS_STATUS_SUCCESS) {
      printIt("Error in CUBAS gemmBatched.\n");
      return status;
   }

   cudaThreadSynchronize();

   cudaError e = cudaGetLastError();
   if (e != cudaSuccess) {return e;}

   status = cublasDestroy(handle);
   if (status != CUBLAS_STATUS_SUCCESS) {
      printIt("Error in CUBAS handle destroy.\n");
      return status;
   }

   e = cudaFree(A_ptr);
   if (e != cudaSuccess) {return e;}
   e = cudaFree(B_ptr);
   if (e != cudaSuccess) {return e;}
   e = cudaFree(C_ptr);
   if (e != cudaSuccess) {return e;}

   return CUBLAS_STATUS_SUCCESS;
}
