/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

/* Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication and is exactly the same as
 * Chapter 7 of the programming guide.
 * It has been written for clarity of exposition to illustrate various CUDA
 * programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication.
 *
 * CUBLAS provides high-performance matrix multiplication.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h>
#include <error_checker.h>
// includes, kernels
#include <matrixMul_kernel.cuh>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char** argv);
void randomInit(float*, int);
void printDiff(float*, float*, int, int);

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

#define cutilSafeCallNoSync(err)     __cudaSafeCallNoSync(err, __FILE__, __LINE__)
#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)
#define cutilSafeThreadSync()        __cudaSafeThreadSync(__FILE__, __LINE__)
#define cutilCheckError(err)         __cutilCheckError   (err, __FILE__, __LINE__)
#define cutilCheckMsg(msg)           __cutilGetLastError (msg, __FILE__, __LINE__)
#define cutilCheckMsgAndSync(msg)    __cutilGetLastErrorAndSync (msg, __FILE__, __LINE__)
#define cutilSafeMalloc(mallocCall)  __cutilSafeMalloc   ((mallocCall), __FILE__, __LINE__)
#define cutilCondition(val)          __cutilCondition    (val, __FILE__, __LINE__)
#define cutilExit(argc, argv)        __cutilExit         (argc, argv)

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
   if( cudaSuccess != err) {
      fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error %d: %s.\n",
               file, line, (int)err, cudaGetErrorString( err ) );
      exit(-1);
   }
}

inline void __cutilCheckError( int err, const char *file, const int line )
{
   if( err != 0 ) {
      fprintf(stderr, "%s(%i) : CUTIL CUDA error.\n",
              file, line);
      exit(-1);
   }
}

inline void __cutilGetLastError( const char *errorMessage, const char *file, const int line )
{
   cudaError_t err = cudaGetLastError();
   if( cudaSuccess != err) {
      fprintf(stderr, "%s(%i) : cutilCheckMsg() CUTIL CUDA error : %s : (%d) %s.\n",
              file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
      exit(-1);
   }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int
main(int argc, char** argv)
{
    runTest(argc, argv);

    exit(EXIT_SUCCESS);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char** argv)
{
    cudaSetDevice( gpuGetMaxGflopsDeviceId() );

    // set seed for rand()
    srand(2006);

    // allocate host memory for matrices A and B
    unsigned int size_A = WA * HA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*) malloc(mem_size_A);
    unsigned int size_B = WB * HB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*) malloc(mem_size_B);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);

    // allocate device memory
    float* d_A;
    cutilSafeCall(cudaMalloc((void**) &d_A, mem_size_A));
    float* d_B;
    cutilSafeCall(cudaMalloc((void**) &d_B, mem_size_B));

    // copy host memory to device
    cutilSafeCall(cudaMemcpy(d_A, h_A, mem_size_A,
                              cudaMemcpyHostToDevice) );
    cutilSafeCall(cudaMemcpy(d_B, h_B, mem_size_B,
                              cudaMemcpyHostToDevice) );

    // allocate device memory for result
    unsigned int size_C = WC * HC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* d_C;
    cutilSafeCall(cudaMalloc((void**) &d_C, mem_size_C));

    // allocate host memory for the result
    float* h_C = (float*) malloc(mem_size_C);
    
    // create and start timer
//    unsigned int timer = 0;
    StopWatchInterface *timer = NULL; // timer object
    cutilCheckError(sdkCreateTimer(&timer));
    cutilCheckError(sdkStartTimer(&timer));

    // setup execution parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(WC / threads.x, HC / threads.y);

    // execute the kernel
    matrixMul<<< grid, threads >>>(d_C, d_A, d_B, WA, WB);

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

    // copy result from device to host
    cutilSafeCall(cudaMemcpy(h_C, d_C, mem_size_C,
                              cudaMemcpyDeviceToHost) );

    // stop and destroy timer
    cutilCheckError(sdkStopTimer(&timer));
    printf("Processing time: %f (ms) \n", sdkGetTimerValue(&timer));
    cutilCheckError(sdkDeleteTimer(&timer));

    // compute reference solution
    float* reference = (float*) malloc(mem_size_C);
    computeGold(reference, h_A, h_B, HA, WA, WB);

    // check result
//    int res = cutCompareL2fe(reference, h_C, size_C, 1e-6f);
//    printf("Test %s \n", (1 == res) ? "PASSED" : "FAILED");
//    if (res!=1) printDiff(reference, h_C, WC, HC);

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);
    cutilSafeCall(cudaFree(d_A));
    cutilSafeCall(cudaFree(d_B));
    cutilSafeCall(cudaFree(d_C));

    cudaThreadExit();
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = rand() / (float)RAND_MAX;
}

void printDiff(float *data1, float *data2, int width, int height)
{
  int i,j,k;
  int error_count=0;
  for (j=0; j<height; j++) {
    for (i=0; i<width; i++) {
      k = j*width+i;
      if (data1[k] != data2[k]) {
         printf("diff(%d,%d) CPU=%4.4f, GPU=%4.4f n", i,j, data1[k], data2[k]);
         error_count++;
      }
    }
  }
  printf(" nTotal Errors = %d n", error_count);
}

