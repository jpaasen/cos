/**
* Part of the RealTimeCapon (RTC) library. 
*
* Author: Jon Petter Aasen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#pragma once

#include "ICapon.h"
#include <Complex.h>
#include <printit.h>

class Capon : ICapon<Complex<float>, float>
{
private:
   Complex<float>* x_gpu;     // data vectors
   Complex<float>* R_gpu;     // covariance matrices
   Complex<float>* Ria_gpu;   // solutions
   Complex<float>* a_gpu;     // right hand sides
   Complex<float>* z_gpu;     // amplitude
   Complex<float>* w_gpu;     // weights

   static const int init_N    = 1;//80000; // Initial value for N
   static const int init_Npx  = 1;//80000; // Initial value for Npx
   static const int init_M    = 1;//64;    // Initial value for M
   static const int init_L    = 1;//32;    // Initial value for L

   int cur_N;     // current number of samples (N) supported by buffer
   int cur_Npx;   // current number of pixels (Npx) supported by buffer
   int cur_M;     // current number of data sample elements (M) supported by buffer
   int cur_L;     // current number of subarray elements (L) supported by buffers

   size_t x_buffer_size;
   size_t R_buffer_size;
   size_t Ria_buffer_size;
   size_t a_buffer_size;
   size_t z_buffer_size;
   size_t w_buffer_size;

   size_t total_buffersize; // the sum of the six variables above

   /** 
   * Init buffers based on N, Npx, M and L.
   * N is the number of input samples, Npx is the number of output pixels,
   * M is the number of elements per input sample and L is the subarray size.
   * Sets the object-variables:
   *   cur_N
   *   cur_M
   *   cur_L
   *   total_buffersize
   * 
   * Returns: total size of required buffers
   **/
   size_t initBufferSize(int N, int Npx, int M, int L);

   /**
   * Allocates GPU buffers based on values given to initBufferSize.
   * Before doing so, it checks that the required memory is available.
   * 
   * Returns: cudaError code
   **/
   int checkAndInitGpuBuffers();

   /** Free GPU buffers **/
   void freeBuffers();

   /**
   * Returns true if total_memory is larger than available device memory on current device, or if total_memory is 0.
   * Otherwise it returns false.
   **/
   bool largerThanMaxSupportedSize(size_t total_memory);

   /**
   * Internal function for calculating Capon weights and output for an image segment
   * Returns cudaError code.
   **/
   int getCaponSegment(
      Complex<float>* &z,   // output amplitude per pixel
      Complex<float>* &w,   // output weights per pixel
      Complex<float>* &R,   // buffer holding the resulting covariance matrices
      Complex<float>* &x,   // buffer holding data vectors
      float &d,				      // diagonal loading factor
      int &L,			         // number of spatial sublengths
      int &Yavg,			      // number of samples averaged in time
      int &M,			         // number of data elements
      int &Nx,			         // number of data vectors in azimuth
      int &Nz,			         // number of data vectors in range
      int &Nb			         // dimension of beamspace
      // TODO: add suport for custom beamspace matrix
      );

   /**
   * Multiply the Npx matrices in A and B. Where A[i] is M*K and B[i] is K*N. C[i] is M*N.
   * opA and opB are operation codes defined in CUBLAS for no operation, transpose and complex conjugated transpose. 
   * CUBLAS_OP_N = 0, CUBLAS_OP_T = 1, CUBLAS_OP_C = 2.
   * lda, ldb and ldc is the leading dimesion of A, B and C as they are saved in memory (before op() is performed).
   * duplicateA and duplicateB can set to true if A or B only contain 1 matrix that is going to be multiplied with Npx matrices. 
   **/
   int batchMultiplyMatrices(
      Complex<float>* A, int opA, 
      Complex<float>* B, int opB, 
      Complex<float>* C, 
      int M, int K, int N, int Npx, 
      int lda, int ldb, int ldc,
      bool duplicateA, bool duplicateB);

public:
   Capon(void);
   ~Capon(void);

   int getCapon(
      Complex<float>* &z,   // output amplitude per pixel
      Complex<float>* &w,   // output weights per pixel
      Complex<float>* &R,   // buffer holding the resulting covariance matrices
      Complex<float>* &x,   // buffer holding data vectors
      float &d,				      // diagonal loading factor
      int &L,			         // number of spatial sublengths
      int &Yavg,			      // number of samples averaged in time
      int &M,			         // number of data elements
      int &Nx,			         // number of data vectors in azimuth
      int &Nz,			         // number of data vectors in range
      int &Nb			         // dimension of beamspace
      // TODO: add suport for custom beamspace matrix
      );	
};
