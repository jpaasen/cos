#ifndef _UHDU_KERNEL_
#define _UHDU_KERNEL_

#include "uhdu.h"

extern __shared__ cuComplex AtoU[]; // Init with UHDU_NUMBER_OF_THREADS * m * sizeof(cuComplex) 
// TODO: init with (m/2)*(m/2)*(32/m)*sizeof(cuComplex) bytes. 
// We can do this when we support only saving active sub-matrix per iteration. The active sub-matrix is at moste (m/2)*(m/2) large.
// A needs to be loaded row by row and is used one time. Therefor it can be loaded from global memory when needed.

// Idea: When diagonal threads finish, can they do forward-backward solving, based on calculated elements?

/**
* Calculates the U'DU decomposition of all dim-m matrices A(i) in A. 
* U(i) is strictly upper-triangular (1 on the diagonal) and is writen back to A. 
* D(i) is saved on the diagonal of U(i) since the diagonal of U is implicite 1.
* 
* Supported values for m are 1 to 32. 
* Best performance when m is a power of 2.
*
* For p <= 3 use closed-form invers kernels.
*
* TODO: Work in progress to have multiple matrices in one thread block.
**/
__global__ void uhdu(cuComplex* A,			// row-major order [Ai1; Ai2; ... Aim;]! 
					 const uint m,			// number of rows (and columns) per matrix
					 const uint mm,			// number of elements per matrix
					 const uint n,			// number of matrices in A
					 const uint rowStrid,	// distance in global memory between rows in a given matrix A(i)
					 const uint matStrid)	// distance in global memory between an equal index in two matrices
{
	//__shared__ cuComplex AtoU[SHARED_MEM];			// shared buffer holding one row of Ai (can hold rows from multiple small matrices)

	const uint blockId = blockIdx.x + blockIdx.y*gridDim.x;	// current group of matrices
	const uint colId = threadIdx.x;							// linear column index for the current matrix group
	const uint matId = threadIdx.y;							// local linear matrix index

	const uint globalMatId = blockId * blockDim.y + matId;	// global linear index of current matrix
	const uint nMatElem = globalMatId * matStrid;			// number of elements in global memory before the current matrix

	const uint sharedRowStrid = m;							// distance between rows in shared memory
	const uint sharedMatStrid = m * blockDim.y;				// distance between an equal index in two matrices in shared memory

	const uint j = colId; // row index
	uint i;

	if (globalMatId < n)// && colId < m) // note: last test should always be true and can be omitted
	{
		// make local pointers to shared and global memory relative to current matrix
		cuComplex* AtoUL			= &AtoU[matId*sharedMatStrid];
		cuComplex* AL			= &A[nMatElem];

		// load matrix into shared memory and sync threads
		//for (i = 0; i < m; ++i)
		//{
		//	AtoUL[i*sharedRowStrid + j] = AL[rowStrid*i + j];
		//}
		//__syncthreads();

		//
		// calculate the uhdu decomposition for each elemnt i,j in column j where i <= j.
		//
		cuComplex upperColSum;
		uint idxI, idxJ, idxK;
		uint k;

		for (i = 0; i <= j; ++i)
		{
			// load one matrix row into shared memory and sync threads
			AtoUL[j] = AL[j];
			__syncthreads();


			upperColSum = make_cuComplex(0.0f, 0.0f);

			if (i < j) { // WARNING! The code depends on low-level knowledg about conditional branching where the else is know to execute before the if block.
				
				for (k = 0; k < i; ++k)
				{
					idxI = k*sharedRowStrid + i;// (k,i)
					idxJ = k*sharedRowStrid + j;// (k,j)
					idxK = k*sharedRowStrid + k;// (k,k)
					upperColSum = cuCaddf(upperColSum, cuCmulf(cuCmulf(cuConjf(AtoUL[idxI]), AtoUL[idxJ]), AtoUL[idxK]));
				}
				
				idxJ = i*sharedRowStrid + j; // (i,j)
				idxI = i*sharedRowStrid + i; // (i,i)
				AtoUL[idxJ] = cuCdivf( cuCsubf(AtoUL[idxJ], upperColSum), AtoUL[idxI]);
				// WARNING: A[idxI] is allready calculated since the else part is executed before the if. This can change in future cuda versions.

			} else { // diagonal element

				for (k = 0; k < i; ++k)
				{
					idxI = k*sharedRowStrid + i; // (k,i)
					idxJ = k*sharedRowStrid + k; // (k,k)
					upperColSum = cuCaddf(upperColSum, cuCmulf(cuCmulf(AtoUL[idxI], cuConjf(AtoUL[idxI])), AtoUL[idxJ]));
				}

				idxI = i*sharedRowStrid + i; // (i,i)
				AtoUL[idxI] = cuCsubf(AtoUL[idxI], upperColSum);
			}
			__syncthreads(); // sync to make prev row visible for all threads in this block
		}
		
		// 
		// Save AtoU to global memory
		//
		for (i = 0; i < m; ++i)
		{
			AL[rowStrid*i + j] = AtoUL[i*sharedRowStrid + j];
		}
	}
}

int uhdu(cuComplex* A, const uint m, const uint n)
{
	dim3 grid(1, 1, 1);
	
	uint numOfThreadBlocksRequired = (n-1)/(UHDU_NUMBER_OF_THREADS / m) + 1;
	
	if (numOfThreadBlocksRequired < MAX_CUDA_GRID_SIZE_XYZ)
	{
		grid.x = numOfThreadBlocksRequired;
	} else {
		grid.x = MAX_CUDA_GRID_SIZE_XYZ;
		grid.y = ((numOfThreadBlocksRequired-1) / 65535) + 1;
	}
	dim3 block(m, UHDU_NUMBER_OF_THREADS / m, 1);

	size_t sharedMem = UHDU_NUMBER_OF_THREADS * m * sizeof(cuComplex);

	// execute decomp kernel
	uhdu<<<grid, block, sharedMem>>>(A, m, m*m, n, m, m*m);

	return cudaGetLastError();
}


#endif