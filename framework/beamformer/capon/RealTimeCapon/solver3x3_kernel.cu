#include "solver3x3_kernel.cuh"

#define N 3
#define NN 9

#define BLOCK_SIZE_X 192//256
#define BLOCK_SIZE_Y 1

#include <vector_types.h>
#include <vector_functions.h>

#include <cuComplex.h>

#include <cutil_math.h> // for vector operations


// Utilities for handeling 3x1 vectors with complex values
//struct __device_builtin__ __builtin_align__(24) cuComplex3
struct cuComplex3
{
   cuComplex x, y, z;
};

inline __host__ __device__ cuComplex3 make_cuComplex3(cuComplex x, cuComplex y, cuComplex z) {
   cuComplex3 c; c.x = x; c.y = y; c.z = z; return c;  
}

inline __host__ __device__ cuComplex3 cross(cuComplex3 a, cuComplex3 b)
{ 
   return make_cuComplex3(cuCmulf(a.y,b.z) - cuCmulf(a.z,b.y), cuCmulf(a.z,b.x) - cuCmulf(a.x,b.z), cuCmulf(a.x,b.y) - cuCmulf(a.y,b.x)); 
}

inline __host__ __device__ cuComplex dot(cuComplex3 a, cuComplex3 b)
{ 
   return cuCmulf(a.x, cuConjf(b.x)) + cuCmulf(a.y, cuConjf(b.y)) + cuCmulf(a.z, cuConjf(b.z));
}

inline __host__ __device__ cuComplex inner(cuComplex3 a, cuComplex3 b)
{ 
   return cuCmulf(a.x, b.x) + cuCmulf(a.y, b.y) + cuCmulf(a.z, b.z);
}

inline __host__ __device__ cuComplex3 operator*(float b, cuComplex3 a)
{
   return make_cuComplex3(b * a.x, b * a.y, b * a.z);
}

inline __host__ __device__ cuComplex3 operator*(cuComplex b, cuComplex3 a)
{
   return make_cuComplex3(cuCmulf(b, a.x), cuCmulf(b,a.y), cuCmulf(b,a.z));
}

/**
* Kernel solving 3x3 sets of linear equations using Cramer's rule
*  
*  x      = solutions
*  A      = matrices
*  b      = left-hand sides
*  stride = distance between elements in a single matrix (not in use)
*	batch  = number of sets
**/
__global__ void solve3x3_kernel(cuComplex* x, const cuComplex* A, const cuComplex* b, const int stride, const int batch) 
{
   //__shared__ cuComplex Ai[BLOCK_SIZE_X*NN];
   //__shared__ cuComplex bi[BLOCK_SIZE_X*N];

   /*const int index_at_block_start   = blockDim.x * blockIdx.x;
   const int local_index            = threadIdx.x;
   const int index                  = index_at_block_start + local_index;*/
   const int index = blockDim.x * blockIdx.x + threadIdx.x;

   if (index < batch) {

      // load values from global to local memory // An attempt to remove poor global memory access pattern.
      /*const int n_threads  = BLOCK_SIZE_X; // TODO: Handle last block! Might have less threads.
      const int threadId   = threadIdx.x;
      int n_elements = BLOCK_SIZE_X*NN;
      A += index_at_block_start*NN;
      b += index_at_block_start*N;
      for (int i = threadId; i < n_elements; i += n_threads) {
         Ai[i] = A[i];
      }
      n_elements = BLOCK_SIZE_X*N;
      for (int i = threadId; i < n_elements; i += n_threads) {
         bi[i] = b[i];
      }*/

      // extract 3 colums to form a 3x3 matrix
      const int mem_bias_m = NN*index;
      cuComplex3 col0 = make_cuComplex3(A[mem_bias_m    ], A[mem_bias_m + 1], A[mem_bias_m + 2]); // TODO: Memory access is no good!
      cuComplex3 col1 = make_cuComplex3(A[mem_bias_m + 3], A[mem_bias_m + 4], A[mem_bias_m + 5]);
      cuComplex3 col2 = make_cuComplex3(A[mem_bias_m + 6], A[mem_bias_m + 7], A[mem_bias_m + 8]);

      //const int mem_bias_m = NN*local_index;
      //cuComplex3 col0 = make_cuComplex3(Ai[mem_bias_m + 0], Ai[mem_bias_m + 1], Ai[mem_bias_m + 2]);
      //cuComplex3 col1 = make_cuComplex3(Ai[mem_bias_m + 3], Ai[mem_bias_m + 4], Ai[mem_bias_m + 5]);
      //cuComplex3 col2 = make_cuComplex3(Ai[mem_bias_m + 6], Ai[mem_bias_m + 7], Ai[mem_bias_m + 8]);

      // calc inverse of A
      cuComplex detMatrix = inner(col0, cross(col1,col2));

      cuComplex a = make_cuComplex(0.0f, 0.0f);
      if (cuCabsf(detMatrix) != 0) {

         a = cuCdivf(make_cuComplex(1.0f,0.0f), detMatrix);//1 / detMatrix; //invertible matrix!

         cuComplex3 invRow0 = a * cross(col1, col2);
         cuComplex3 invRow1 = a * cross(col2, col0);
         cuComplex3 invRow2 = a * cross(col0, col1);

         // read b from global memory to registers
         //const int mem_bias_v = local_index*N;
         //cuComplex3 b_reg = make_cuComplex3(bi[mem_bias_v], bi[mem_bias_v + 1], bi[mem_bias_v + 2]);
         const int mem_bias_v = N*index;
         cuComplex3 b_reg = make_cuComplex3(b[mem_bias_v], b[mem_bias_v + 1], b[mem_bias_v + 2]);

         // calc x and write it back to global memory
         x[mem_bias_v    ] = inner(invRow0, b_reg);
         x[mem_bias_v + 1] = inner(invRow1, b_reg);
         x[mem_bias_v + 2] = inner(invRow2, b_reg);

      } else {
         x[index] = make_cuComplex(0.5f, 0.0f); // debug value // in matlab we get here. So there is somthing wrong with the assignment of col0, col1 and col2.
         return; // we might want to exit here, or do something else
      }
   }
}

int solve3x3(cuComplex* x, const cuComplex* A, const cuComplex* b, const int batch) 
{
   if (batch > 0) {
      dim3  threads(BLOCK_SIZE_X, 1, 1);
      dim3  grid((batch-1)/BLOCK_SIZE_X + 1, 1, 1);

      // set cache configuration to maximize L1 cache
      cudaFuncSetCacheConfig(solve3x3_kernel, cudaFuncCachePreferL1);

      int stride = 1; // distance between each element in a matrix in A.

      // execute the kernel
      solve3x3_kernel<<< grid, threads>>>(x, A, b, stride, batch);

      return cudaGetLastError();
   
   } else {
      return cudaErrorInvalidValue;
   }
}
