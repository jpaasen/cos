#ifndef _MATRIXINVERSION_KERNEL_H_
#define _MATRIXINVERSION_KERNEL_H_

#define N 3

#define BLOCK_SIZE_X 256
#define BLOCK_SIZE_Y 1

#include <vector_types.h>
#include <vector_functions.h>

#include <cutil_math.h> // for vector operations


__global__ void matrixInversion3x3(float3* matrices, const unsigned int numberOfMatrices) 
{

  const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

  // extract 3 colums to form a 3x3 matrix
  float3 col0 = matrices[index];
  float3 col1 = matrices[index + numberOfMatrices]; 
  float3 col2 = matrices[index + numberOfMatrices*2];

  float detMatrix = dot(col0, cross(col1,col2));

  float a = 0;
  if (detMatrix != 0) 
	  a = 1 / detMatrix; //invertible matrix!
  else
	  ; // we might want to exit here, or do something else

  float3 invRow0 = a * cross(col1, col2);
  float3 invRow1 = a * cross(col2, col0);
  float3 invRow2 = a * cross(col0, col1);

  matrices[index]					   = invRow0;
  matrices[index + numberOfMatrices]   = invRow1;
  matrices[index + numberOfMatrices*2] = invRow2;

}


/**
* Kernel supported by matlab parallel computing toolbox
*
*	n = number of matrices
*
*	Matrices are saved liearly in column order with a strid of n for 
*		
*		Linear matrix indexing
*		[1 4 7;
*		 2 5 8;
*		 3 6 9]
*
*		Format of parameter float* matrices
*		[m11 ... mn1 m12 ... mn2 ...... m19 ... mn9]'
*
**/
__global__ void matlabMatrixInversion3x3(float* matrices, const unsigned int n) 
{

  const unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

  // extract 3 colums to form a 3x3 matrix
  float3 col0 = make_float3(matrices[index      ], matrices[index +   n], matrices[index + 2*n]);
  float3 col1 = make_float3(matrices[index + 3*n], matrices[index + 4*n], matrices[index + 5*n]);
  float3 col2 = make_float3(matrices[index + 6*n], matrices[index + 7*n], matrices[index + 8*n]);
  //float3 col1 = matrices[index + numberOfMatrices]; 
  //float3 col2 = matrices[index + numberOfMatrices*2];

  float detMatrix = dot(col0, cross(col1,col2));

  float a = 0;
  if (detMatrix != 0) {
	  a = 1 / detMatrix; //invertible matrix!
  } else {
	  matrices[index] = 0.5f; // debug value // in matlab we get here. So there is somthing wrong with the assignment of col0, col1 and col2.
	  return; // we might want to exit here, or do something else
  }

  float3 invRow0 = a * cross(col1, col2);
  float3 invRow1 = a * cross(col2, col0);
  float3 invRow2 = a * cross(col0, col1);

  matrices[index      ] = invRow0.x;
  matrices[index +   n] = invRow1.x;
  matrices[index + 2*n] = invRow2.x;
  matrices[index + 3*n] = invRow0.y;
  matrices[index + 4*n] = invRow1.y; 
  matrices[index + 5*n] = invRow2.y;
  matrices[index + 6*n] = invRow0.z;
  matrices[index + 7*n] = invRow1.z;
  matrices[index + 8*n] = invRow2.z;

  //matrices[index]					   = invRow0;
  //matrices[index + numberOfMatrices]   = invRow1;
  //matrices[index + numberOfMatrices*2] = invRow2;

}

#endif
