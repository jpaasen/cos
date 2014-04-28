/**
* Part of the RealTimeCapon (RTC) library. 
*
* Author: Jon Petter Aasen & Jo Inge Buskenes
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/

#include "capon_kernel.cuh"
#include "CudaUtils.h"

#include <cuda_runtime.h>
#include <printit.h>

// run power capon instead of amplitude capon // TODO: Make this change available at runtime!
//#define POWER_CAPON

__global__ void values_gpu(cuComplex* b, float value, int NL, int L, bool beamspace)
{
   const int tid = (blockIdx.x * blockDim.y * gridDim.y) + (blockDim.y * blockIdx.y) + threadIdx.y;

   if (tid < NL) {
      if (beamspace) {
         b[tid] = (tid%L)? make_cuComplex(0.0f, 0.0f) : make_cuComplex(value, 0.0f);
      } else {
         b[tid] = make_cuComplex(value, 0.0);
      }
   }
}

int values(cuComplex* b, float value, int NL, int L, bool beamspace) 
{
   int max_threads_per_block = 128;

   dim3 block(1, max_threads_per_block, 1);
   dim3 grid = cuUtilsGetGrid(NL, block.y);

   printIt("Ones: grid=(%d,%d), block=(%d,%d)\n", grid.x, grid.y, block.x, block.y);

   values_gpu<<<grid,block>>>(b, value, NL, L, beamspace);

   return cudaGetLastError();
}

// Kernel for generating the beamspace butler matrix on the gpu
// B is outputted in column major order.
// Npx matrices are generated.
__global__ void butler_matrix_gpu(cuComplex* B, int Nb, int L, int Npx)
{
   const int matrix = (blockIdx.x * blockDim.y * gridDim.y) + (blockDim.y * blockIdx.y) + threadIdx.y; 

   if (matrix < Npx) {

      const int b = threadIdx.x;
      int beam = b;

      if (b > Nb/2) { // code for laying out beams symmetric around zero. If Nb is even, one more beam is positioned in positive angular direction.
         beam += (L - 2*(Nb/2) - 1);
         if (Nb&1 == 0) beam += 1;
      }

      float res_L = rsqrtf(float(L));
      float angel_factor = 2.0f * PI * beam / L;
      float l_f = 0.0f;

      for (int l = 0; l < L; ++l) {

         float sin_ptr, cos_ptr;
         sincosf(l_f*angel_factor, &sin_ptr, &cos_ptr);

         cuComplex value = make_cuComplex(res_L*cos_ptr, -res_L*sin_ptr);

         B[Nb*L*matrix + Nb*l + b] = value;

         l_f += 1.0f;
      }
   }
}

int butler_matrix(cuComplex* B, int Nb, int L, int Npx)
{
   int max_threads_per_block = 128;
   if (Nb > max_threads_per_block) {
      max_threads_per_block = 512;
   }

   if (Nb > max_threads_per_block) {
      printIt("Max leading dimension for Butler matrix is %d.\n", max_threads_per_block);
      return cudaErrorLaunchFailure;
   }

   dim3 block(Nb, max_threads_per_block/Nb);
   dim3 grid = cuUtilsGetGrid(Npx, block.y);

   butler_matrix_gpu<<<grid, block>>>(B, Nb, L, Npx);

   cudaThreadSynchronize();

   return cudaGetLastError();
}


#define x(vectorNum, index) x[((M)*(vectorNum)) + (index)]
#define xi(vectorNum, index) xi[((M)*(vectorNum)) + (index)]

/**
* Calculates the subarray sum of data vector x_i, i = [0, M-1].
* s_i = sum_{k=i}^{i+M-L+1}(x_k).
* N is the number of data vectors.
* 
* Not tested!!!
**/
__global__ void sum_subarrays_kernel(cuComplex*       s,       // Subarray sums (output)
   const cuComplex* x,       // Data vector
   const int        M,       // Number of channels
   const int        L,       // Subarray size
   const int		   Yavg,    // Time averaging samples (2*Yavg + 1 in total)
   const int		   Ny,		// Number of samples in one range line
   const int        N)       // Number of pixels
{
   extern __shared__ float dyn_shared_mem[];

   cuComplex* x_smem = (cuComplex*)dyn_shared_mem;
   // can easily add more use of shared memory here. Use registers first!

   const int pixel           = (blockIdx.x * blockDim.y * gridDim.y) + (blockDim.y * blockIdx.y) + threadIdx.y;
   const int local_pixel     = threadIdx.y;
   const int L_idx           = threadIdx.x;

   if (pixel < N) {

      if (L_idx < L) {

         // Read data vector x from global memory to shared
         int i;
         int yavg_bias = ((pixel/(Ny-2*Yavg))*2 + 1)*(Yavg*M);
         for(i=0; i<M/L; ++i ) {
            x_smem[local_pixel*M + L*i + L_idx] = x[pixel*M + L*i + L_idx + yavg_bias]; 
         }
         if( L_idx < M%L ) { // read rest elements
            x_smem[local_pixel*M + L*i + L_idx] = x[pixel*M + L*i + L_idx + yavg_bias];
         }

         __syncthreads(); // all threads inside a block has to be finished reading data before reducing x

         // Reduce the data vector from length M to L
         cuComplex sum = make_cuComplex(0.0f, 0.0f);
         for(int k=0; k < M-L+1; ++k) {
            sum = cuCaddf(sum, x_smem[local_pixel*M + L_idx+k]);
         }

         s[pixel*L + L_idx] = sum;
      }
   }

}

/**
* Calculate beamformer output from subarray sums. 
*
* Launch using: block(L, MAX_THREADS_PER_BLOCK/L) and the cuUtilsGetGrid-function
**/
__global__ void calc_output_from_subarray_sums_kernel(cuComplex*         z,         // Pixel amplitude (output)
   cuComplex*         w,         // Weights used (output)
   const cuComplex*   s,         // subarray sums
   const cuComplex*   Ria,       // R^(-1)a
   const int          oL,        // original number of subarrays
   const int          L,         // subarray size
   const int          Yavg,      // Time averaging
   const int          Ny,        // Number of pixels in range 
   const int          N,         // Number of pixels 
   const bool         beamspace) // If true, the weight vector is normalized using the first element only, else w is norm with the element sum.
{
   extern __shared__ float dyn_shared_mem[];

   cuComplex* w_smem = (cuComplex*)dyn_shared_mem;

   // can easily add more use of shared memory here. Use registers first!

   const int pixel        = (blockIdx.x * blockDim.y * gridDim.y) + (blockDim.y * blockIdx.y) + threadIdx.y;
   const int local_pixel  = threadIdx.y;
   const int L_idx        = threadIdx.x;

   cuComplex sum;
   cuComplex L_element;

   if (pixel < N) {

      // Read Ria from global memory to shared
      L_element = Ria[pixel*L + L_idx];
      w_smem[local_pixel*L + L_idx] = L_element;

      __syncthreads();

      // Find the normalization coefficient (a^h R^-1 a)
      if (beamspace) {
         cuComplex temp = w_smem[local_pixel*L];
         //sum = make_cuComplex(temp.x, temp.y);
         sum = make_cuComplex(sqrtf(float(oL))*temp.x, sqrtf(float(oL))*temp.y);
      } else {
         sum = make_cuComplex(0.0f, 0.0f);
         for( int l = 0; l < L; ++l ) {
            sum = cuCaddf(sum, w_smem[local_pixel*L+l]);
         }
      }
      __syncthreads();

      // Compute normalized w
      L_element = cuCdivf(L_element, sum); // keeping it in register

      // Write the output weights back to global memory
      w[pixel*L+L_idx] = L_element; // TODO: Read-after-write move down...

      // Read subarray sums from global memory to register
      s += L*(pixel + Yavg*(1 + 2*(pixel/(Ny-2*Yavg)))); // offset s. If Yavg > 0 we need an extra offset
      sum = s[L_idx];
      //printf("%d (%f,%f)\n", L_idx, sum.x, sum.y);

      // Compute output from each subarray channel
      w_smem[local_pixel*L+L_idx] = cuCmulf(sum, cuConjf(L_element));//L_element); // TODO: Have observed large numerical errors here!
      __syncthreads();

      // Compute the output amplitude by summing each subarray channel
      if( L_idx == 0 ) {
         sum = make_cuComplex(0.0f, 0.0f);
         for(int l=0; l<L; ++l) {
            sum = cuCaddf(sum, w_smem[local_pixel*L + l]);
         }

         // Write pixel amplitude to global memory
         z[pixel] = sum; // subarray sums are allready normalized
      }
   }
}

__global__ void power_capon_kernel(cuComplex*       z,         // Pixel amplitude (output)
   cuComplex*       w,         // Weights used (output)
   const cuComplex* x,         // Data vector
   const cuComplex* Ria,       // R^(-1)a
   const int        M,         // Number of channels
   const int        L,         // Subarray size
   const int		    Yavg,	   // Time averaging samples (2*Yavg + 1 in total)
   const int		    Ny,		   // Number of samples in one range line
   const int        N)         // Number of pixels
{
   extern __shared__ float dyn_shared_mem[];

   cuComplex* w_smem = (cuComplex*)dyn_shared_mem;
   cuComplex* x_smem = &w_smem[blockDim.y*L];

   // can easily add more use of shared memory here. Use registers first!

   const int pixel					= (blockIdx.x * blockDim.y * gridDim.y) + (blockDim.y * blockIdx.y) + threadIdx.y;
   const int local_pixel			= threadIdx.y;
   const int L_idx					= threadIdx.x;

   cuComplex sum;
   cuComplex L_element;

   if (pixel < N) {

      if (L_idx < L) { // use the L first threads to load weights and data

         // Read Ria from global memory to shared
         L_element = Ria[pixel*L + L_idx];
         w_smem[local_pixel*L + L_idx] = L_element;

         __syncthreads();

         // Find the normalization coefficient (a^h R^-1 a)
         sum = make_cuComplex(0.0f, 0.0f);
         for( int l = 0; l < L; ++l ) {
            sum = cuCaddf(sum, w_smem[local_pixel*L + l]);
         }

         // Compute normalized w
         L_element = cuCdivf(L_element, sum); // keeping it in register

         // Add hamming weight to lower sidelobes // Test by Jon Petter 23.10.2012, strange results
         //if (L > 1) {
         //	L_element = cuCmulf( L_element, make_cuComplex(0.54f - 0.46f*cosf(6.28f*(float(L_idx)/float(L-1))), 0.0f) );
         //}

         // Write the output weights back to shared and global memory
         w[pixel*L+L_idx] = L_element;
         __syncthreads();
         w_smem[local_pixel*L+L_idx] = L_element;

      }

      int K = M-L+1;
      __syncthreads(); // __syncthreads only effects threads that are currently active in the given conditional-block!

      if (L_idx < K) {

         // Read data vector x from global memory to shared
         int i;
         int yavg_bias = ((pixel/(Ny-2*Yavg))*2 + 1)*(Yavg*M);
         for(i=0; i<M/K; ++i ) {
            x_smem[local_pixel*M + K*i + L_idx] = x[pixel*M + K*i + L_idx + yavg_bias]; 
         }
         if( L_idx < M%K ) { // read rest elements
            x_smem[local_pixel*M + K*i + L_idx] = x[pixel*M + K*i + L_idx + yavg_bias];
         }

         __syncthreads(); // all threads inside a block has to be finished reading data before output is calculated

         // calc subarray outputs
         sum = make_cuComplex(0.0f, 0.0f);
         for(int k = 0; k < L; ++k) {
            sum = cuCaddf(sum, cuCmulf(cuConjf(w_smem[local_pixel*L + k]) , x_smem[local_pixel*M + L_idx+k]));
         }

         __syncthreads(); // since the output below is saved to x_smem we need to sync (could we have used w_smem?)
         //x_smem[local_pixel*K+L_idx] = sum; // amplitude capon
         //x_smem[local_pixel*K + L_idx] = make_cuComplex( cuCabsf(sum), 0.0f ); // power capon using abs value
         x_smem[local_pixel*K + L_idx] = cuCmulf( sum, cuConjf(sum) ); // power capon

         __syncthreads();  

         // Compute the output amplitude by summing each subarray channel
         if( L_idx == 0 ) {
            float sum_f = 0.0f;
            //sum = make_cuComplex(0.0f, 0.0f);
            for(int l = 0; l < K; ++l) {
               //sum = cuCaddf(sum, x_smem[local_pixel*K + l]);
               sum_f += x_smem[local_pixel*K + l].x;
            }

            // Write pixel amplitude to global memory
            //z[pixel] = make_cuComplex(sum.x/K, sum.y/K);
            z[pixel] = make_cuComplex(sqrtf(sum_f)/K, 0.0f);
         }
      }
   }
}


__global__ void amplitude_capon_kernel(cuComplex*       z,         // Pixel amplitude (output)
   cuComplex*       w,         // Weights used (output)
   const cuComplex* x,         // Data vector
   const cuComplex* Ria,       // R^(-1)a
   const int        M,         // Number of channels
   const int        L,         // Subarray size
   const int		  Yavg,		 // Time averaging samples (2*Yavg + 1 in total)
   const int		  Ny,			 // Number of samples in one range line
   const int        N)         // Number of pixels
{
   extern __shared__ float dyn_shared_mem[];

   cuComplex* w_smem = (cuComplex*)dyn_shared_mem;
   cuComplex* x_smem = &w_smem[blockDim.y*L];

   // can easily add more use of shared memory here. Use registers first!

   const int pixel           = (blockIdx.x * blockDim.y * gridDim.y) + (blockDim.y * blockIdx.y) + threadIdx.y;
   const int local_pixel     = threadIdx.y;
   const int L_idx           = threadIdx.x;

   cuComplex sum;
   cuComplex L_element;

   if (pixel < N) {

      if (L_idx < L) {

         // Read Ria from global memory to shared
         L_element = Ria[pixel*L+L_idx];
         w_smem[local_pixel*L+L_idx] = L_element;

         __syncthreads();

         // Find the normalization coefficient (a^h R^-1 a)
         sum = make_cuComplex(0.0f, 0.0f);
         for( int l = 0; l < L; ++l ) {
            sum = cuCaddf(sum, w_smem[local_pixel*L+l]);
         }

         // Compute normalized w
         L_element = cuCdivf(L_element, sum); // keeping it in register

         // Add hamming weight to lower sidelobes // Test by Jon Petter 23.10.2012, strange results, Tested again 12.12.2012, no improvement invivo
         //if (L > 1) {
         //	L_element = cuCmulf( L_element, make_cuComplex(0.54f - 0.46f*cosf(6.28f*(float(L_idx)/float(L-1))), 0.0f) );
         //}

         // Write the output weights back to global memory
         w[pixel*L+L_idx] = L_element;

         // Read data vector x from global memory to shared
         int i;
         int yavg_bias = ((pixel/(Ny-2*Yavg))*2 + 1)*(Yavg*M);
         for(i=0; i<M/L; ++i ) {
            x_smem[local_pixel*M + L*i + L_idx] = x[pixel*M + L*i + L_idx + yavg_bias]; 
         }
         if( L_idx < M%L ) { // read rest elements
            x_smem[local_pixel*M + L*i + L_idx] = x[pixel*M + L*i + L_idx + yavg_bias];
         }

         __syncthreads(); // all threads inside a block has to be finished reading data before reducing x

         // Reduce the data vector from length M to L
         sum = make_cuComplex(0.0f, 0.0f);
         for(int k=0; k<M-L+1; ++k ) {
            sum = cuCaddf(sum, x_smem[local_pixel*M + L_idx+k]);
         }

         __syncthreads(); // since the output below is saved to x_smem we need to sync (could we have used w_smem?)

         // Compute output from each subarray channel
         x_smem[local_pixel*L+L_idx] = cuCmulf(sum, cuConjf(L_element)); // using element in registers instead
         __syncthreads();

         // Compute the output amplitude by summing each subarray channel
         if( L_idx == 0 ) {
            sum = make_cuComplex(0.0f, 0.0f);
            for(int l=0; l<L; ++l) {
               sum = cuCaddf(sum, x_smem[local_pixel*L + l]);
               //sum = cuCaddf( sum, cuCmulf(x_smem[local_pixel*L + l], cuConjf(x_smem[local_pixel*L + l])) ); // Testing Power Capon - jpaasen 16.10.2012
               //sum = cuCaddf( sum, make_cuComplex(cuCabsf(x_smem[local_pixel*L + l]), 0.0f) ); // Testing Power Capon - jpaasen 16.10.2012
            }

            // Write pixel amplitude to global memory
            z[pixel] = make_cuComplex(sum.x/(M-L+1), sum.y/(M-L+1));
         }
      }
   }
}

int getCaponOutputFromSubarraySums(cuComplex*         z_gpu,      // Pixel amplitude (output)
   cuComplex*         w_gpu,      // Weights used (output)
   const cuComplex*   s_gpu,      // Subarray sums vector
   const cuComplex*   Ria_gpu,    // R^(-1)a
   const int          oL,         // original subarray size before beamspace transformation
   const int          L,          // Subarray size after transformation
   const int          Yavg,       // Temporal averaging
   const int          Ny,         // Number of pixels in range
   const int          N,          // Total number of pixels
   const bool         beamspace)  // If true, capon weights are normalized with the first element of Ria only, not the sum as in element space
{
   int threads_per_block = 128;

   dim3 block, grid;
   block.x = L;
   size_t shared_memory;

   while (threads_per_block > 1 && threads_per_block >= L) { 

      block.y = threads_per_block/L;
      grid = cuUtilsGetGrid(N, block.y);

      printIt("getCaponOutputFromSubarraySums: grid=(%d,%d), block=(%d,%d)\n", grid.x, grid.y, block.x, block.y);

      // Specify the amount of shared memory needed per block
      shared_memory = block.y*L*sizeof(cuComplex);

      if (shared_memory < 48e3) {
         break;
      }
      threads_per_block /= 2;
   }

   if( shared_memory > 48e3 ) {
      printIt("L=%d is too large for all the data (%d) to be held in shared memory (max 48kB).\n", L, (int)(shared_memory/1000));
      return cudaErrorLaunchFailure;
   }
   calc_output_from_subarray_sums_kernel<<<grid, block, shared_memory>>>(z_gpu, w_gpu, s_gpu, Ria_gpu, oL, L, Yavg, Ny, N, beamspace);
   return cudaGetLastError();
}

/**
* Calculates w = Ria/a^HRia, and applies the weights on x -> z = w*x
**/
int getCaponOutput(cuComplex*       z_gpu,   // Pixel amplitude (output)
   cuComplex*       w_gpu,   // Weights used (output)
   const cuComplex* x_gpu,   // Data vector
   const cuComplex* Ria_gpu, // R^(-1)a
   const int        M,       // Number of channels
   const int        L,       // Subarray size
   const int			Yavg,		// Time averaging size (In total 2*Yavg+1 samples)
   const int	   	Ny,		// Number of samples in one range line
   const int        N)       // Number of pixels
{
   int threads_per_block = 128;

   // Run the kernel
#ifdef POWER_CAPON
   // run power capon (incoherent sum of sub arrays)
   int K = M-L+1;
   dim3 block_pow(K, threads_per_block/K, 1);
   dim3 grid_pow = getGrid(N, block_pow.y);
   // Specify the amount of shared memory needed per block
   size_t shared_memory_pow = block_pow.y*(M+L)*sizeof(cuComplex);
   if( shared_memory_pow > 48e3 ) {
      printIt("M=%d is too large for all the data (%d) to be held in shared memory (max 48kB).\n", M, (int)(shared_memory_pow/1000));
      return cudaErrorLaunchFailure;
   }
   if (L <= K) {
      power_capon_kernel<<<grid_pow, block_pow, shared_memory_pow>>>(z_gpu, w_gpu, x_gpu, Ria_gpu, M, L, Yavg, Ny, N);
   } else {
      printIt("Values of L > M/2 is not supported for power capon.");
   }
#else
   // run amplitude capon (coherent sum of subarrays)
   dim3 block, grid;
   block.x = L;
   size_t shared_memory;

   while (threads_per_block > 1 && threads_per_block >= L) { 

      // Launch one block for a set of pixels
      block.y = threads_per_block/L;
      grid = cuUtilsGetGrid(N, block.y);

      printIt("Capon: grid=(%d,%d), block=(%d,%d)\n", grid.x, grid.y, block.x, block.y);

      // Specify the amount of shared memory needed per block
      shared_memory = block.y*(M+L)*sizeof(cuComplex);

      if (shared_memory < 48e3) {
         break;
      }
      threads_per_block /= 2;
   }
   if( shared_memory > 48e3 ) {
      printIt("M=%d and L=%d are too large for all the data (%d) to be held in shared memory (max 48kB).\n", M, L, (int)(shared_memory/1000));
      return cudaErrorLaunchFailure;
   }

   amplitude_capon_kernel<<<grid, block, shared_memory>>>(z_gpu, w_gpu, x_gpu, Ria_gpu, M, L, Yavg, Ny, N);
#endif

   return cudaGetLastError();

}
