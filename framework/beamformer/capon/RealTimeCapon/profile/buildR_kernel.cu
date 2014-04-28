/**
* Part of the RealTimeCapon (RTC) library. 
*
* Author: Jon Petter Aasen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/

#include "buildR_kernel.cuh"
#include "CudaUtils.h"

#include <printit.h>
#include "cudaConfig.h"
#include <cuda_runtime.h>
#include <cutil_math.h>

#define BUILD_R_NUMBER_OF_THREADS 128// per block

#define x(vectorNum, index) x[((M)*(vectorNum)) + (index)]
#define xi(vectorNum, index) xi[((M)*(vectorNum)) + (index)]
#define R(row, col) R[(L*(row)) + (L*L*(matrixVectorId)) + (col)] // Output R in memory using stride 1 for each row ()
//#define R(row, col) R[((L)*(N)*(row)) + ((L)*(matrixVectorId)) + (col)] // Output R in memory using stride N for each row (This will minimize global wrights)

// TODO: Save upper triangle of R linearly in memory
//#define IDX(row,col) (((row)*(L)) + (col) - (((col)*((col)+1))/2)) // formula to lay out upper triangle of R in linear memory (Li + j - (i*(i+1)/2))
//#define Ri(col) Ri[((L)*(blockIdx.y)) + (col)] // macro preveously used to save R in local memory 
//extern __shared__ cuComplex Ri[]; // init during kernel launch. Size must be: L*blockDim.y*sizeof(cuComplex)

/**
* Launch this kernel with
*	blockdim.x == L
*	blockdim.y == number of vectors/matrices per thread block 
*   blockIdx.x is the fastest moving index, then blockIdx.y
* 
*  gridDim.x = Nx
*  gridDim.y = (Ny-2*Yavg-1)/blockDim.y + 1
* 
* Supported values: 
*   L <= MAX_THREADS_PER_BLOCK
*   M and Yavg such that (M+L)*MAX_THREADS_PER_BLOCK/L + (2*Yavg*M) < MAX_SHARED_MEMORY_PER_BLOCK
*
* The columns of R is saved to memory using strid 1.
* Can easially be change to output R using row sride of N, by changing the R macro above?
*
**/

__global__ void buildR_kernel(
                              const cuComplex* x,		// complex data vectors
                              cuComplex* R,				// complex sampel covariance matrices in column-major-order
                              const float d,			   // diagonal loading factor
                              const uint L,				// length of sub-arrays
										const uint Yavg,			// number samples used in time averaging (2*Yavg + 1 in total)
                              const uint K,				// number of sub-arrays (M-L+1)
                              const uint M,				// data vector dim
                              const uint Nx,          // number of datasamples (usually in azimuth) 
                              const uint Ny           // number of datasamples (usually in range)
                              )		
{
   extern __shared__ float dyn_shared_mem[];

   cuComplex* xi        = (cuComplex*)dyn_shared_mem;                // buffer to hold data
   cuComplex* firstRowR = (cuComplex*)&xi[M*(blockDim.y + 2*Yavg)];  // buffer to hold first row of R to conjugate-transpose it on the fly, also used to hold diagonal elements
   // can easily add more use of shared memory here. Remember: Use registers first!

   const uint columnId        = threadIdx.x;									      // j  // column managed by current thread
   const uint localMatVecId   = threadIdx.y;                               // n (local)  // x and R processed by current thread
   const uint threadId        = (blockDim.x * threadIdx.y) + threadIdx.x;  // &R[i,j] (local)  // linear index of current thread inside its block

   const uint matrixVectorIdX = blockIdx.x;                                         // x  // index of current x and R in x-dir
   const uint matrixVectorIdY = (blockDim.y * blockIdx.y) + threadIdx.y;            // n+Yavg  // index of current x and R in y-dir
   const uint matrixVectorId  = ((Ny-2*Yavg) * matrixVectorIdX) + matrixVectorIdY;  // &R[x,n+Yavg] // linear index of current x and R (fast-moving in y-dir)

#if defined(MATH_ONLY_GLOBAL) || defined(MATH_ONLY_SHARED)
   cuComplex tmp;
   float flag = (float)(int)(M/90.0); //Always zero, but the compiler doesn't know that
#endif
#ifdef MATH_ONLY_SHARED
   float f0 = (float)(int)(M/70.0); //Always zero, but the compiler doesn't know that
   float f1 = (float)(int)(M/80.0)+1; //Always one, but the compiler doesn't know that
#endif

   if (matrixVectorIdY < Ny-2*Yavg) {

      if (columnId < L) {

         // Read x from global memory into shared memory
         uint n_vectors = blockDim.y;
         if (blockIdx.y == gridDim.y - 1) {                                // Last gridline in y-dir must be handled separately
            n_vectors = n_vectors - (gridDim.y*blockDim.y - (Ny-2*Yavg));  // gridDim.y*blockDim.y - Ny is always >= 0 && < blockDim.y
         }
         uint tot_num_elem          = M * (2*Yavg + n_vectors);	// Total number of elements that needs to be read from global memory by current block
         uint n_threads_per_block   = L * n_vectors;			      // Number of threads in current block

         for (uint k = threadId; k < tot_num_elem; k += n_threads_per_block) {
            // cuConjf has been added so that R is really outputted in column-major order (even though the code might says otherwise)
            // The reason is simple: The Nvidia solver wants it that way
#ifndef MATH_ONLY_GLOBAL
   #ifndef MATH_ONLY_SHARED
         xi[k] = cuConjf( x[M*(Ny*blockIdx.x + blockDim.y*blockIdx.y) + k] );
   #else
         tmp = cuConjf( x[M*(Ny*blockIdx.x + blockDim.y*blockIdx.y) + k] );
   #endif
#else
   #ifndef MATH_ONLY_SHARED
         xi[k] = cuConjf( make_cuComplex(M*(Ny*blockIdx.x + blockDim.y*blockIdx.y) + k, 0.0f));
   #else
         tmp = cuConjf( make_cuComplex(M*(Ny*blockIdx.x + blockDim.y*blockIdx.y) + k, 0.0f));
   #endif
#endif

         }
         __syncthreads(); // sync to make elements in shared memory available to all threads in this block

         // Build R and write one element (blockDim.y rows per thread block) to global memory per iteration

         cuComplex sum = make_cuComplex(0.0f, 0.0f);

         uint col = columnId;
         uint row;

         // calc first row of R
         for(row = 0; row < K; ++col, ++row) // Average over sub-arrays
         {
            for (uint n = 0; n < 2*Yavg+1; ++n) { // Time averaging
#ifndef MATH_ONLY_SHARED
               sum = cuCaddf(sum, cuCmulf(xi(localMatVecId+n, row), cuConjf(xi(localMatVecId+n, col)))); // TODO: potential for bank conflicts here?
#else
               sum = cuCaddf(sum, cuCmulf(tmp, cuConjf(make_cuComplex(f0, f1)))); // TODO: potential for bank conflicts here?
#endif
            }
         }
#ifndef MATH_ONLY_GLOBAL
         R(0, columnId) = sum;
#else
         if( 1.0==sum.x*flag )
            R(0, columnId) = sum;
#endif
         // subsequent rows are calculated by updating previous result for current thread (sliding window over sub-arrays)
         // Let dying threads copy values to fill the whole of R. 
         // Need a buffer to hold first row of R. If col == 0 we copy from this buffer, else we slide
#ifndef MATH_ONLY_SHARED
         firstRowR[L*localMatVecId + columnId] = sum;
#else
         if( 1.0==sum.x*flag )
            firstRowR[L*localMatVecId + columnId] = sum;
#endif
         __syncthreads(); // when L is larger than a warp we have to sync

         float trace = 0.0f;
         if (columnId == 0) {
            trace = sum.x;
         }

         // col and row are in this code pointing at the x(row)x(col)^H we want to subtract from sum
         for (col = columnId, row = 0; row < L-1; ++col, ++row) {
            if (col >= L-1) { // current thread has finished its sub-diagonal and is ready for a new one on the transposed side of R
              
               col = 0, --col; // DANGER! Based on uint overflow, -1+1 == MAX_UINT+1 == 0. Only use col+1 after this statement!

#ifndef MATH_ONLY_SHARED
               sum = cuConjf( firstRowR[L*localMatVecId + row + 1] );
#else
               sum = cuConjf( make_cuComplex(f1, f0) );
#endif

            } else {
               for (uint n = 0; n < 2*Yavg+1; ++n) { // Time averaging
#ifndef MATH_ONLY_SHARED
                  sum = cuCaddf(sum, cuCsubf(
                     cuCmulf(xi(localMatVecId + n, row + K),   cuConjf(xi(localMatVecId + n, col + K))),
                     cuCmulf(xi(localMatVecId + n, row),       cuConjf(xi(localMatVecId + n, col)))
                     ));
#else
                  sum = cuCaddf(sum, cuCsubf(
                     cuCmulf( make_cuComplex(f1, f0),  cuConjf( make_cuComplex(f0, f1))),
                     cuCmulf( make_cuComplex(f1, f0),  cuConjf( make_cuComplex(f1, f0)))
                     ));
#endif

               }
            }
#ifndef MATH_ONLY_GLOBAL
            R(row+1, col+1) = sum; // output calculated row of R to global memory
#else
         if( 1.0==sum.x*flag )
            R(row+1, col+1) = sum;
#endif
         __syncthreads();
            if (columnId == 0) { // main diagonal thread
               // save diagonal by overwriting used elements in the firstRowR buffer
#ifndef MATH_ONLY_SHARED
               firstRowR[L*localMatVecId + col + 1] = sum;
#else
               if( 1.0==sum.x*flag )
                  firstRowR[L*localMatVecId + col + 1] = sum;
#endif
               trace += sum.x;
            }
         }

         // sync so that we can access diagonal elements across threads
         __syncthreads();
#ifndef MATH_ONLY_SHARED
         float diagonalElem = firstRowR[L*localMatVecId + columnId].x;
#else
         float diagonalElem = f1;
#endif
			__syncthreads();

         // broadcast trace to all threads
         if (columnId == 0) {
#ifndef MATH_ONLY_SHARED
            firstRowR[L*localMatVecId].x = trace;
#else
         if( 1.0==sum.x*flag )
            firstRowR[L*localMatVecId].x = trace;
#endif
         }
         __syncthreads();
#ifndef MATH_ONLY_SHARED
         trace = firstRowR[L*localMatVecId].x;
#else
         trace = f0;
#endif

         // Add diagonal loading factor
         float inv_L = 1.0f/L;
#ifndef MATH_ONLY_GLOBAL
         R(columnId, columnId).x = diagonalElem + d * trace * inv_L;
#else
         if( 1.0==sum.x*flag )
            R(columnId, columnId).x = diagonalElem + d * trace * inv_L;
#endif

         // Normalize result with 1/(M-L+1)
         // Note: There is no need to normalize R in order to get right capon weights

      }
   }
}

/**
* Kernel that calculates the covariance matrix R using one thread per element.
* Use this if L < 10 approx.
*
* This kernel can be launched using a 3D block of threads blockDim=(L,L,N). With the z-index representing a given matrix.
**/
__global__ void buildR_full_kernel(const cuComplex*  x,		 // complex data vectors
                                   cuComplex*        R,		 // complex sampel covariance matrices in column-major-order.
                                   const float       d,		 // diagonal loading factor
                                   const int         L,		 // length of sub-arrays
                                   const int         Yavg,	 // number samples used in time averaging (2*Yavg + 1 in total)
                                   const int         K,		 // number of sub-arrays (M-L+1)
                                   const int         M,		 // data vector dim
                                   const int         Nx,     // number of datasamples (usually in azimuth) 
                                   const int         Ny,     // number of datasamples (usually in range)
                                   const int         subarrayStrid)  // strid between each subarray (1 and L is supported) If L is used, M and K must be selected with care
{
   extern __shared__ float dyn_shared_mem[];

   cuComplex* xi    = (cuComplex*)dyn_shared_mem;                 // buffer to hold data
   cuComplex* diagR = (cuComplex*)&xi[M*(blockDim.z + 2*Yavg)];   // buffer to hold diagonal elements 
   // can easily add more use of shared memory here. Remember: Use registers first!

   const int column       = threadIdx.x;
   const int row          = threadIdx.y;
   const int elementIdx   = row*blockDim.x + column;
   const int matrixSize   = blockDim.x * blockDim.y;
   const int localMatrix  = threadIdx.z;
   const int threadId     = matrixSize*localMatrix + elementIdx;
   
   const int matrixIdxX = blockIdx.x;
   const int matrixIdxY = blockIdx.y*blockDim.z + threadIdx.z;

   const int matrix = ((Ny-2*Yavg) * matrixIdxX) + matrixIdxY;

   if (matrixIdxY < (Ny - 2*Yavg)) {

      // Read x from global memory into shared memory
      int n_matrices = blockDim.z;                                        // Number of matrices calculated by the current block
      if (blockIdx.y == gridDim.y - 1) {                                   // Last gridline in y-dir must be handled separately
         n_matrices = n_matrices - (gridDim.y*blockDim.z - (Ny-2*Yavg));   // gridDim.y*blockDim.x - Ny is always >= 0 && < blockDim.y
      }
      int tot_num_elem          = M * (2*Yavg + n_matrices);              // Total number of elements that needs to be read from global memory by current block
      int n_threads_per_block   = matrixSize * n_matrices;		            // Number of threads in current block

      // Offset x to right position in global memory with respect to current block.
      x += M*(Ny*matrixIdxX + blockDim.z*blockIdx.y); 

      for (int k = threadId; k < tot_num_elem; k += n_threads_per_block) {
         // cuConjf has been added so that R is really outputted in column-major order (even though the code might says otherwise)
         // The reason is simple: The Nvidia solver wants it that way
         xi[k] = cuConjf( x[k] );
      }
      __syncthreads(); // sync to make elements in shared memory available to all threads in this block

      // Calc covariance coefficients
      cuComplex sum = make_cuComplex(0.0f, 0.0f);
      cuComplex x1, x2;
      
      for (int n = 0; n < 2*Yavg+1; ++n) { // Time averaging
         for (int l = 0; l < K; ++l) { // Subarray averaging
            x1 = xi(localMatrix+n, row+(l*subarrayStrid)); // TODO: potential for bank conflicts here?
            x2 = xi(localMatrix+n, column+(l*subarrayStrid));
            sum = cuCaddf(sum , cuCmulf( x1, cuConjf(x2) )); 
         }
      }
      
      R[L*L*matrix + elementIdx] = sum; // write output to global memory

      // Adaptive diagonal loading
      if (column == row) {
         diagR[L*localMatrix + column] = sum;
      }
      __syncthreads();
      cuComplex trace = make_cuComplex(0.0f, 0.0f);
      if (elementIdx == 0) {
         for (int i = 0; i < L; ++i) {
            trace = cuCaddf(trace, diagR[L*localMatrix + i]);
         }
         diagR[L*localMatrix] = trace;
      }
      __syncthreads();
      trace = diagR[L*localMatrix];
      if (column == row) {
         float inv_L = 1.0f/L;
         R[L*L*matrix + elementIdx].x = sum.x + d*trace.x*inv_L;
      }
   }
}

dim3 getGrid(const uint &Nx, const uint &Ny, const uint &NperBlock, const uint &Yavg) {
   dim3 grid( Nx, ( (Ny - 2*Yavg - 1)/NperBlock ) + 1, 1 );
   return grid;
}

int build_R(const cuComplex* x, cuComplex* R, const float d, const uint L, const uint Yavg, const uint M, const uint Nx, const uint Ny)
{ 
   if (L > BUILD_R_NUMBER_OF_THREADS) {
      printIt("Error buildR: L is too large for current launch setup\n");
      return cudaErrorLaunchFailure;
   }

   // TODO:
   // block.y should be adaptively set to 6(warps/block) * (32(warpsize)/L)
   // then lowered accordingly if shared memory consuption gets to high.
   // uint n_threads_per_block = 512;

   dim3 blockR(L, BUILD_R_NUMBER_OF_THREADS/L, 1); // TODO: Maybe blockR.x should be rounded up to N*warp_size? Then we need to include somthing like if (blockIdx.x < L) in our kernel
   dim3 gridR = getGrid(Nx, Ny, blockR.y, Yavg);

   // Calculate shared memory usage
   size_t shared_mem_size = blockR.y*M;   // add shared memory for x
   shared_mem_size += L*blockR.y;         // add shared memory for firstRowR buffer
	shared_mem_size += 2*Yavg*M;           // add extra shared memory if Yavg > 0;
	shared_mem_size *= sizeof(cuComplex);  // add size of cuComplex in bytes

   // print grid-block config and shared mem
   printIt("grid=(%d,%d), block=(%d,%d), shared mem=%dKB/block\n", gridR.x, gridR.y, blockR.x, blockR.y, shared_mem_size/1000);

   if (shared_mem_size > 48000) {
      // too much shared memory in use per block
      printIt("Error buildR: M, L and Yavg are in a range that makes use of too much shared memory per block.");
      return cudaErrorLaunchFailure;
   }

   uint K = M - L + 1;

   buildR_kernel<<<gridR, blockR, shared_mem_size>>>(x, R, d, L, Yavg, K, M, Nx, Ny);

   return cudaGetLastError();
}

int build_R_full(const cuComplex* x, cuComplex* R, const float d, const int L, const int Yavg, const int M, const int Nx, const int Ny, const int subarrayStrid)
{ 
   int threads_per_block = 128;

   if (L > 22) { // gradually increase block size if L is large. Up to L_max == 32.
      threads_per_block = 1024;
   } else if (L > 16) {
      threads_per_block = 512;
   } else if (L > 11) {
      threads_per_block = 256;
   }

   if (L*L > threads_per_block) {
      printIt("Error buildR: L is too large for current launch setup (L > 32, L was %d)\n", L);
      return cudaErrorLaunchFailure;
   }

   dim3 blockR(L, L, threads_per_block/(L*L));
   dim3 gridR = getGrid(Nx, Ny, blockR.z, Yavg);

   // Calculate shared memory usage
   size_t shared_mem_size = blockR.z*M;   // add shared memory for x
   shared_mem_size += L*blockR.z;         // add shared memory for diagR buffer
	shared_mem_size += 2*Yavg*M;           // add extra shared memory if Yavg > 0;
	shared_mem_size *= sizeof(cuComplex);  // add size of cuComplex in bytes

   // print grid-block config and shared mem
   printIt("grid=(%d,%d), block=(%d,%d), shared mem=%dKB/block\n", gridR.x, gridR.y, blockR.x, blockR.y, shared_mem_size/1000);

   if (shared_mem_size > 48000) {
      // too much shared memory in use per block
      printIt("Error buildR: M, L and Yavg are in a range that makes use of too much shared memory per block.");
      return cudaErrorLaunchFailure;
   }

   uint K = M/subarrayStrid - (L-1)/subarrayStrid; // nuber of "possible" subarrays

   buildR_full_kernel<<<gridR, blockR, shared_mem_size>>>(x, R, d, L, Yavg, K, M, Nx, Ny, subarrayStrid);

   return cudaGetLastError();
}
