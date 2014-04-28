/**
* Part of the RealTimeCapon (RTC) library. 
*
* Author: Jon Petter Aasen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#include "Solver.h"
#include <cuda_runtime.h>
#include "printit.h"

#include "solver1x1_kernel.cuh"
#include "solver3x3_kernel.cuh"

Solver::Solver(SolverType type)
{
   solverType = type;
}

Solver::~Solver(void)
{
}

int Solver::solve(
   Complex<float>* &x_in,        // buffer holding the solutions
   Complex<float>* &A_in,  // buffer holding matrices
   Complex<float>* &b_in,  // buffer holding the left sides
   int &N,                 // size of each linear system
   int &batch,             // number of linear systems
   bool &x_on_gpu,         // true if x should remain on the gpu
   bool &A_on_gpu,         // true if R is already on the gpu
   bool &b_on_gpu          // true if b is already on the gpu
   )
{
   if (N < 1 || N > 72) {
      printIt("Error in Solver: Min N is 1 and max N is 72.\n");
      return cudaErrorLaunchFailure;
   }

   if (batch < 1) {
      printIt("Erroro in Solver: batch must be larger than 0.\n");
      return cudaErrorLaunchFailure;
   }

   cuComplex *x = (cuComplex*)x_in;
   cuComplex *A = (cuComplex*)A_in;
   cuComplex *b = (cuComplex*)b_in;

   // TODO: Call zfsolve_batch in solve.h
   cudaError e;

   cuComplex* x_gpu;
   cuComplex* A_gpu;
   cuComplex* b_gpu;

   size_t x_buffer_size = N*batch*sizeof(cuComplex);
   size_t A_buffer_size = N*N*batch*sizeof(cuComplex);
   size_t b_buffer_size = N*batch*sizeof(cuComplex);

   if (x_on_gpu) {
      x_gpu = x; 
   } else { // TODO: make a function for this...
      e = cudaMalloc<cuComplex>(&x_gpu, x_buffer_size);
      if (e != cudaSuccess) return e;
      e = cudaMemcpy((void*)x_gpu, (void*)x, x_buffer_size, cudaMemcpyHostToDevice);
      if (e != cudaSuccess) return e;
   }
   if (A_on_gpu) {
      A_gpu = (cuComplex*) A; // not good, but what to do?
   } else {
      e = cudaMalloc<cuComplex>(&A_gpu, A_buffer_size);
      if (e != cudaSuccess) return e;
      e = cudaMemcpy((void*)A_gpu, (void*)A, A_buffer_size, cudaMemcpyHostToDevice);
      if (e != cudaSuccess) return e;
   }
   if (b_on_gpu) {
      b_gpu = (cuComplex*) b;
   } else {
      e = cudaMalloc<cuComplex>(&b_gpu, b_buffer_size);
      if (e != cudaSuccess) return e;
      e = cudaMemcpy((void*)b_gpu, (void*)b, b_buffer_size, cudaMemcpyHostToDevice);
      if (e != cudaSuccess) return e;
   }

   // TODO:
   // For small matrices one should use a kernel evaluating the invers (one thread per matrix)
   // There is one such kernel for 3x3 in the RealTimeCapon folder, it should be possible to make kernels for 4x4 and 5x5 as well.
   // Tip from Nvidia: up to 10x10 it should be faster to go with a 1-matrix-per-thread appraoch.
   // This seems to be done in zfsolve_batch. There is different kernels beeing called depending on N.

   if (N == 1) {
      e = (cudaError) solve1x1(x_gpu, A_gpu, b_gpu, batch);
   } else if (N == 3 && this->solverType == Solver::DIRECT) {
      e = (cudaError) solve3x3(x_gpu, A_gpu, b_gpu, batch);
   } else {
      e = (cudaError) zfsolve_batch(A_gpu, b_gpu, x_gpu, N, batch);
   }

   if (e != cudaSuccess) return e; // TODO: What about allocated gpu memory if kernel fail?

   if (!x_on_gpu) {
      e = cudaMemcpy((void*)x, (void*)x_gpu, x_buffer_size, cudaMemcpyDeviceToHost);
      if (e != cudaSuccess) return e;
      e = cudaFree((void*)x_gpu);
      if (e != cudaSuccess) return e;
   }
   if (!A_on_gpu) {
      e = cudaFree((void*)A_gpu);
      if (e != cudaSuccess) return e;
   }
   if (!b_on_gpu) {
      e = cudaFree((void*)b_gpu);
      if (e != cudaSuccess) return e;
   }

   return e;
}
