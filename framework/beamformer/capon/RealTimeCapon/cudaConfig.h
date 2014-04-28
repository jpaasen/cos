#ifndef _CUDA_CONFIG_
#define _CUDA_CONFIG_

#define MAX_CUDA_GRID_SIZE_XYZ 65535

#if __CUDA_ARCH__ >= 200
	#define MAX_M 32
	#define MAX_MN 1024
	#define MAX_NUMBER_OF_THREADS 32//128
	#define MAX_N_PER_BLOCK 1				// max 16x16 matrices per thread block
	#define SHARED_MEM 1024//4096			// 4096 can hold 4 single precision complex matrices of size 32. 1024 can hold 1.
	#define SHARED_MEM_VEC 32//128			// 128 can hold 4 single precision complex vectors of size 32. 32 can hold 1.
#else
	#define MAX_NUMBER_OF_THREADS 112
	#define MAX_N_PER_BLOCK 7				// max 16x16 matrices per thread block
	#define SHARED_MEM 1792					// can hold 7 single precision complex matrices of size 16
	#define SHARED_MEM_VEC 112				// can hold 7 single precision complex vectors of size 16
	#define MAX_M 16
	#define MAX_N 16
	#define MAX_MN 256
#endif

#endif
