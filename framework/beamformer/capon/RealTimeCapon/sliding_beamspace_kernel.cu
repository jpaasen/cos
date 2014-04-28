/**
* Part of the RealTimeCapon (RTC) library. 
*
* Author: Jon Petter Aasen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/

#include "sliding_beamspace_kernel.cuh"
#include "CudaUtils.h"

#include <printit.h>
#include <cuda_runtime.h>
#include <algorithm>

/**
* Helper function to calculate the sliding beamspace transformation.
*
* x must be a pointer to a M long data vector, where N_L = M - L + 1.
* For each of the N_L sub arrays we output Nb coeffs to global memory x_bs.
* In addition x_bs is summed across subarrays and outputted to s_bs.
* Offset x, x_bs and s_bs with respect to current sample and threadgroup before calling this function.
* b is the local beam id.
*
* x_bs s_bs are normalized (multiplied) with res_L.
**/
__inline__ __device__ void slideDFT(cuComplex*  x,       // input data samples
                                    cuComplex*  x_bs,    // transformed output
                                    cuComplex*  s_bs,    // transformed output summed across subarrays
                                    const int   &b,      // selected beam index [0,Nb-1]
                                    const int   &L,      // subarray size   
                                    const float &res_L,  // 1/sqrt(L) or 1/L
                                    const int   &N_L,    // number of subarrays
                                    const int   &Nb)     // number of beams
{
   cuComplex factor;
   float cos_ptr, sin_ptr;

   cuComplex sum        = make_cuComplex(0.0f, 0.0f);
   cuComplex totalSum   = make_cuComplex(0.0f, 0.0f);
   s_bs[b]              = make_cuComplex(0.0f, 0.0f);

   int beam = b;
   if (b > Nb/2) { // code for laying out beams symmetric around zero. If Nb is even, one more beam is positioned in positive angular direction.
      beam += (L - 2*(Nb/2) - 1);
      if (Nb&1 == 0) beam += 1;
   }

   __syncthreads();

   float l_f = 0.0f;
   float angle_factor = 2.0f * PI * beam / L; 

   // start with one complete transformation of the first sub array
   for (int l = 0; l < L; ++l) {

      sincosf(l_f*angle_factor, &sin_ptr, &cos_ptr);
      factor = make_cuComplex(cos_ptr, -sin_ptr);

      sum = cuCaddf( sum, cuCmulf(x[l], factor) );

      l_f += 1.0f;
   }

   x_bs[b] = make_cuComplex(res_L*sum.x, res_L*sum.y); // write transformed data to memory
   totalSum = sum;

   // init sliding multiplication factors
   sincosf(angle_factor, &sin_ptr, &cos_ptr);
   factor = make_cuComplex(cos_ptr, sin_ptr);

   // for the remaining N_L - 1 subarrays we slide.
   for (int l = 1; l < N_L; ++l) {
   
      sum = cuCmulf( cuCsubf( cuCaddf(sum, x[l+L-1]), x[l-1]), factor );

      x_bs[l*Nb + b] = make_cuComplex(res_L*sum.x, res_L*sum.y);//sum;
      totalSum = cuCaddf(totalSum, sum);
   }
   atomicAdd(&s_bs[b].x, res_L*totalSum.x); // ! only suported in cc. > 2.0
   atomicAdd(&s_bs[b].y, res_L*totalSum.y);
   __syncthreads();
}

/**
* Calculates the sliding beamspace/DFT transformation for a set of symmetric beams around broadside (angle=0)
* 
* The kernel use a combination of sliding and full computations to balance number of threads per used byte.
* The idea is to let nThreadGroupsPerSample cooperate when computing the beamspace transformation for one sample.
* Each threadGroup will make use of the sliding beamspace transform.
*
* Note that data is transformed with a un-normalized Butler matrix, however the subarray sums (s) are normalized with 1/sqrt(L).
**/ 
__global__ void sliding_beamspace_kernel(const cuComplex* x,               // input data samples
                                         cuComplex* x_bs,                  // transformed output data
                                         cuComplex* s,                     // transformed output data summed across subarrays
                                         const int M,                      // Number of elements in one data sample (x[i:i+M])
                                         const int L,                      // Number of subarrays
                                         const float res_L,                // 1/sqrt(L) or 1/L
                                         const int Nb,                     // Number of beams in beamspace
                                         const int Nx,                     // Number of samples in azimuth
                                         const int Ny,                     // Number of samples in range
                                         const int nThreadGroupsPerSample, // .
                                         const int avgNSubarraysPerGroup,  // .
                                         const int nRestSubarrays,         // .
                                         const int nBlocks)                // Number launched thread blocks
{
   extern __shared__ float dyn_shared_mem[];

   const int sample = (blockIdx.x * blockDim.y * gridDim.y) + (blockDim.y * blockIdx.y) + threadIdx.y; 

   if (sample < Nx*Ny) {

      const int localSample     = threadIdx.y;
      const int sampleThreadId  = threadIdx.x;
      const int threadGroup     = sampleThreadId / Nb;
      const int threadId        = blockDim.x*localSample + sampleThreadId;
      
      const int N_L = M-L+1;

      cuComplex* x_smem = (cuComplex*)dyn_shared_mem;
      cuComplex* s_smem = &x_smem[blockDim.y*M];

      // Load data
      // move pointers to correct location in memory for the current block
      x += M*(sample-threadIdx.y);
      
      const int Nsamples  = (sample/blockDim.y == nBlocks-1) ? (blockDim.y - (nBlocks*blockDim.y - Nx*Ny)) : blockDim.y; // handle last block in grid
      const int Nthreads  = /*(blockDim.x*Nsamples >= 96) ? 96 :*/ blockDim.x*Nsamples;
      const int Nvalues   = M*Nsamples;
      //if (threadId < Nthreads) {
         for (int i = threadId; i < Nvalues; i += Nthreads) {
            x_smem[i] = x[i];
         }
      //}
      //__syncthreads(); // is not needed since shared memory is not used before after sync in slideDFT

      uint nSubArraysToBeProcessed = avgNSubarraysPerGroup;
      //if (threadGroup == nThreadGroupsPerSample - 1) {
      //   nSubArraysToBeProcessed += nRestSubarraysPerGroup; // Handle rest subarrays in the last thread group // TODO: Add one more in several threads instead
      //}
      if (threadGroup < nRestSubarrays) {
         nSubArraysToBeProcessed += 1;
      }

      int threadGroupFac = threadGroup*avgNSubarraysPerGroup;
      if (threadGroup >= nRestSubarrays) {
         threadGroupFac += nRestSubarrays;
      }

      // offset pointers to right memory location
      cuComplex* x_ptr    = x_smem + M*localSample + threadGroupFac;//threadGroup*avgNSubarraysPerGroup;
      cuComplex* x_bs_ptr = x_bs   + Nb*N_L*sample + Nb*threadGroupFac;//threadGroup*avgNSubarraysPerGroup;
      cuComplex* s_bs_ptr = s_smem + Nb*localSample;
      
      const int beam = sampleThreadId % Nb;

      slideDFT(x_ptr, x_bs_ptr, s_bs_ptr, beam, L, res_L, nSubArraysToBeProcessed, Nb);

      if (threadGroup == 0) {
         float inv_N_L = 1.0f/N_L;
         s[sample*Nb + beam] = make_cuComplex( s_bs_ptr[beam].x*inv_N_L, s_bs_ptr[beam].y*inv_N_L);
      }
   }
}

int sliding_beamspace(const cuComplex* x,
                      cuComplex* x_bs,
                      cuComplex* s,  
                      const int M,
                      const int L,
                      const int Nb,
                      const int Nx, 
                      const int Ny)
{
   if (M < 1 || L < 1 || Nb < 1 || Nx < 1 || Ny < 1) {
      printIt("Error in sliding_beamspace: M, L, Nb, Nx or Ny was < 1, M=%d, L=%d, Nb=%d, Nx=%d, Ny=%d\n", M, L, Nb, Nx, Ny);
      return cudaErrorLaunchFailure;
   }

   if (Nx > MAX_CUDA_GRID_SIZE_XYZ || Ny > MAX_CUDA_GRID_SIZE_XYZ) {
      printIt("Error in sliding_beamspace: Nx or Ny is larger than suported size %d, Nx=%d, Ny=%d\n", MAX_CUDA_GRID_SIZE_XYZ, Nx, Ny);
      return cudaErrorLaunchFailure;
   }

   //int max_threads_per_block = 128;
   int max_threads_per_block = 192;
   if (Nb > 22) { // gradually increase block size if L is large. Up to L_max == 32.
      max_threads_per_block = 1024;
   } else if (Nb > 16) {
      max_threads_per_block = 512;
   } else if (Nb > 11) {
      max_threads_per_block = 256;
   }

   int K = M-L+1;
   int n_threadgroups_per_sample, avg_n_subarrays_per_group, n_rest_subarrays;
   dim3 block, grid;
   int nBlocks;
   size_t shared_mem;

   while (max_threads_per_block > 1 && max_threads_per_block >= Nb) {

      n_threadgroups_per_sample = max_threads_per_block*(M+Nb)*sizeof(cuComplex)*MAX_CUDA_BLOCKS_SM / (MAX_CUDA_SMEM_SIZE * Nb) + 1;
      n_threadgroups_per_sample = std::min<int>(n_threadgroups_per_sample, K);
      avg_n_subarrays_per_group = K / n_threadgroups_per_sample;
      n_rest_subarrays          = K % n_threadgroups_per_sample;
      //n_rest_subarrays_per_group = K % avg_n_subarrays_per_group;
      //n_rest_subarrays_per_group = K - (n_threadgroups_per_sample*avg_n_subarrays_per_group);

      block.x = Nb*n_threadgroups_per_sample;
      block.y = max_threads_per_block/(Nb*n_threadgroups_per_sample);
      grid = cuUtilsGetGrid(Nx*Ny, block.y);

      nBlocks = (Nx*Ny-1)/block.y + 1;

      shared_mem = block.y*(M+Nb)*sizeof(cuComplex);

      if (shared_mem < 48e3) {
         break;
      }
      max_threads_per_block /= 2;
   }
   if( shared_mem > 48e3 ) {
      printIt("Error in sliding_beamspace: M=%d, L=%d and Nb=%d are too large for all the data (%d) to be held in shared memory (max 48kB).\n", M, Nb, L, (int)(shared_mem/1000));
      return cudaErrorLaunchFailure;
   }
   if (Nb*n_threadgroups_per_sample > max_threads_per_block) {
      printIt("Error in sliding_beamspace: M, L and Nb is in a range that combined use to much shared memory, M=%d, L=%d, Nb=%d\n", M, L, Nb);
      return cudaErrorLaunchFailure;
   }

   sliding_beamspace_kernel<<<grid, block, shared_mem>>>(
      x, x_bs, s, M, L, rsqrtf(float(L)), Nb, Nx, Ny, 
      n_threadgroups_per_sample, avg_n_subarrays_per_group, n_rest_subarrays, nBlocks);

   return cudaGetLastError();
}

