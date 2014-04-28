#include "CudaUtils.h"

#include <stdlib.h>
#include <stdio.h>
#include <climits>
#include <algorithm>
#include <stdexcept>
#include <sstream>

void cuUtilCheckCudaError(const cudaError &e, const std::string msg) {
   if (e != cudaSuccess) {
      throw std::runtime_error(std::string("CUDA ERROR: ") + cudaGetErrorString(e));
   }
}

void cuUtilCheckAvailableMemory(size_t memSizeWanted) {
   size_t free, total;
   cuUtilCheckCudaError( cudaMemGetInfo(&free, &total) );

   if (free < memSizeWanted) {
      std::stringstream ss;
      ss << "Error: Not enough available GPU memory.\n" << "Needed " << (memSizeWanted/1e6) << "Mb, had " << (free/1e6) << "Mb available.\n" <<
         "Try to divide your problem into smaller pices." <<
         "\nTotal GPU memory is " << (total/1e6) << "Mb.";

      throw std::runtime_error(ss.str());
   }
}

void cuUtilsSafeCall(cudaError err, bool dieOnError)
{
	if (err != cudaSuccess) {
		fprintf(stderr, "%s\n", cudaGetErrorString(err));
		if (dieOnError) {
			exit(-1);
		}
	}
}

void cuUtilSetDevice(bool useMaxNumSM)
{
	int dev_count;
	cudaError e = cudaGetDeviceCount(&dev_count);

   if (e != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(e));
   }

   int min_dev_sm = INT_MAX;
   int max_dev_sm = 0;
   int min_dev = 0;
   int max_dev = 0;

   if (useMaxNumSM) {
      cudaDeviceProp dev_prop;
      int i;
      for (i = 0; i < dev_count; ++i) {
         cudaGetDeviceProperties(&dev_prop, i);

         if (dev_prop.multiProcessorCount > max_dev_sm) {
            max_dev_sm = dev_prop.multiProcessorCount;
            max_dev = i;
         } else if (dev_prop.multiProcessorCount < min_dev_sm) {
            min_dev_sm = dev_prop.multiProcessorCount;
            min_dev = i;
         }
      }

   } else {
      max_dev = dev_count - 1;
   }
	
	e = cudaSetDevice(max_dev);

	if (e != cudaSuccess) {
		printf("%s\n", cudaGetErrorString(e));
	}
}

dim3 cuUtilsGetLinearGridX(int N, int linearBlockDim)
{
   dim3 grid;

   int nGridLinesX = (N-1)/(MAX_CUDA_GRID_SIZE_XYZ*linearBlockDim) + 1;

   grid.x = nGridLinesX;
   grid.z = 1;

   if (nGridLinesX > 1) {
      grid.y = MAX_CUDA_GRID_SIZE_XYZ;
   } else {
      grid.y = ((N-1)/linearBlockDim) + 1;
   }
   return grid;
}

dim3 cuUtilsGetGrid(int N, int blockDimY) {
   return cuUtilsGetLinearGridX(N, blockDimY);
}

void cuUtilGetLinearBlockAndGrid(dim3 &grid, dim3 &block, const int N, const int linearBlockDim, const bool xLayout) {
   block.x = xLayout? linearBlockDim : 1; 
   block.y = xLayout? 1 : linearBlockDim;
   grid.x  = std::min<int>(N/linearBlockDim + 1, MAX_CUDA_GRID_SIZE_XYZ);
   grid.y  = (grid.x - 1)/MAX_CUDA_GRID_SIZE_XYZ + 1;
   if (!xLayout) {
      int tmp = grid.x;
      grid.x = grid.y;
      grid.y = tmp;
   }
}

void cuUtilGetSquareBlockAndGrid(dim3 &grid, dim3 &block, const int M, const int N, const int blockDimX, const int blockDimY) {
   block.x = 16;
   block.y = 16;
   grid.x = (M-1)/block.x + 1;
   grid.y = (N-1)/block.y + 1;
}
