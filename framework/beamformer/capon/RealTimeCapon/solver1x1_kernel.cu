#include "solver1x1_kernel.cuh"

#define BLOCK_SIZE_X 192
#define BLOCK_SIZE_Y 1

#include <vector_types.h>
#include <vector_functions.h>

#include <cutil_math.h> // for vector operations

#include "CudaUtils.h"


/**
* Kernel solving 1x1 sets of linear equations, so its just division...
*  
*  x      = solutions
*  A      = matrices
*  b      = left-hand sides
*	batch  = number of sets
**/
__global__ void solve1x1_kernel(cuComplex* x, const cuComplex* A, const cuComplex* b, const int batch) 
{

   const int index = (gridDim.x*blockDim.x*blockIdx.y) + blockDim.x*blockIdx.x + threadIdx.x;

   if (index < batch) {
      x[index] = cuCdivf(b[index], A[index]);
   }
}

int solve1x1(cuComplex* x, const cuComplex* A, const cuComplex* b, const int batch) 
{
   if (batch > 0) {
      dim3  block, grid;
      cuUtilGetLinearBlockAndGrid(grid, block, batch);

      // set cache configuration to maximize L1 cache
      cudaFuncSetCacheConfig(solve1x1_kernel, cudaFuncCachePreferL1);

      // execute the kernel
      solve1x1_kernel<<<grid, block>>>(x, A, b, batch);

      return cudaGetLastError();
   
   } else {
      return cudaErrorInvalidValue;
   }
}
