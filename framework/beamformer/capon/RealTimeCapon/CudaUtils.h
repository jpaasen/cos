#ifndef _CUDA_UTILS_HEADER_
#define _CUDA_UTILS_HEADER_

#include <cuda_runtime.h>
#include <string>

// Hack to remove errors on CUDA types in VS2010 IntelliSense.
#ifdef _WIN32
#ifndef __CUDACC__
   #include "device_launch_parameters.h"
   
   extern void __syncthreads(void);

   extern float rsqrtf(float x);
   extern void sincosf(float a, float *sptr, float *cptr);

   extern float atomicAdd(float *address, float val);
#endif
#endif

#ifndef MAX_CUDA_GRID_SIZE_XYZ
#define MAX_CUDA_GRID_SIZE_XYZ 65535
#endif

#ifndef MAX_CUDA_SMEM_SIZE
#define MAX_CUDA_SMEM_SIZE 48000
#endif

#ifndef MAX_CUDA_BLOCKS_SM
#define MAX_CUDA_BLOCKS_SM 8
#endif

#ifndef MAX_CUDA_BLOCK_SIZE_XY
#define MAX_CUDA_BLOCK_SIZE_XY 512
#endif

#ifndef PI
#define PI 3.14159265358979323846f
#endif

/* Function that trows exeption upon cuda error */
void cuUtilCheckCudaError(const cudaError &e, const std::string msg="");

/* Function for handling cuda errors. IF dieOnError is true, exit(-1) is called */
void cuUtilsSafeCall(cudaError err, bool dieOnError);

/* Set cuda device. If useMaxNumSM is true the GPU with max number of SMs is used. If useMaxNumSM is false the last GPU is used. */
void cuUtilSetDevice(bool useMaxNumSM=true);

/* Check if available memory is larger than memSizeWanted. If not an exception is thrown. */
void cuUtilCheckAvailableMemory(size_t memSizeWanted); 

/**
* Function for laying out blocks of threads in a 2D grid.
* Fastmoving in grid.y, then grid.x.
**/
dim3 cuUtilsGetGrid(int N, int blockDimY);

void cuUtilGetLinearBlockAndGrid(dim3 &grid, dim3 &block, const int N, const int linearBlockDim=192, const bool xLayout=true);

void cuUtilGetSquareBlockAndGrid(dim3 &grid, dim3 &block, const int M, const int N, const int blockDimX=16, const int blockDimY=16);

#endif
