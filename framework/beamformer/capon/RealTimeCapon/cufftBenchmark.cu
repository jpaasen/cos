#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>

// includes, project
#include <cufft.h>
#include <cutil_inline.h>

#define MAX_CUFFT_ELEMENTS 64000000

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

// forward dec
void runTest(unsigned int numberOfFFTs, unsigned int sizeOfFFTs);
void selectDevice(int argc, char** argv);

int main( int argc, char** argv) 
{
    printf("[Batch FFT test]\n");

	unsigned int numberOfFFTs = 1024;
	if (argc > 1) numberOfFFTs = atoi(argv[1]);

	unsigned int sizeOfFFTs = 16;
	if (argc > 2) sizeOfFFTs = atoi(argv[2]);

	printf("-----------------------------------------------------------------\n");

	selectDevice(argc, argv);

	printf("-----------------------------------------------------------------\n");
	printf("Times = [Plan time, FFT processing time, GPU+CPU time]\n");
	printf("-----------------------------------------------------------------\n");

	runTest(numberOfFFTs, sizeOfFFTs);

	printf("-----------------------------------------------------------------\n");
	
	for (unsigned int j = 0; j <= 2; j++) {
		for (unsigned int i = 5; i < 10; i++) {
			runTest(numberOfFFTs*(2 << i), sizeOfFFTs*(2 << j));
		}
		printf("-----------------------------------------------------------------\n");
	}

	cutilExit(argc, argv);
}

void runTest(unsigned int numberOfFFTs, unsigned int sizeOfFFTs) 
{
	printf("[L, #] = [%3d %7d] ", sizeOfFFTs, numberOfFFTs);

	unsigned int num_elements = numberOfFFTs * sizeOfFFTs;
	unsigned int mem_size = sizeof(cufftComplex) * num_elements;

	printf("%7d KB", mem_size/1000);

    // allocate host memory
	cufftHandle plan;
	cufftComplex *data;

	if (num_elements < MAX_CUFFT_ELEMENTS) {
		cudaMalloc( (void**)&data, mem_size);
	} else {
		cudaMalloc( (void**)&data, mem_size/4);
	}

	clock_t t1 = clock();

	// make timer
	unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

	// create a 1D FFT plan
	if (num_elements < MAX_CUFFT_ELEMENTS) {
		cufftPlan1d(&plan, sizeOfFFTs, CUFFT_C2C, numberOfFFTs);
	} else {
		cufftPlan1d(&plan,  sizeOfFFTs, CUFFT_C2C, numberOfFFTs/4);
	}

	cutilSafeCall(cudaThreadSynchronize());

	cutilCheckError( cutStopTimer( timer));
	float timeValue = cutGetTimerValue( timer);
	timeValue = (num_elements < MAX_CUFFT_ELEMENTS)? timeValue : timeValue*4;
    printf( " %6.2f [ms] ", timeValue);
    cutilCheckError( cutDeleteTimer( timer));

    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

	// use the CUFFT plan to transform the signals in place
	if (num_elements < MAX_CUFFT_ELEMENTS) {
		cufftExecC2C(plan, data, data, CUFFT_FORWARD);
	} else {
		cufftExecC2C(plan,  data, data, CUFFT_FORWARD);
	}

	cutilSafeCall(cudaThreadSynchronize());

	cutilCheckError( cutStopTimer( timer));
	timeValue = cutGetTimerValue( timer);
	timeValue = (num_elements < MAX_CUFFT_ELEMENTS)? timeValue : timeValue*4;
    printf( " %4.2f [ms]", timeValue);
    cutilCheckError( cutDeleteTimer( timer));

	clock_t t2 = clock();
	timeValue = ((t2 - t1))/double(CLOCKS_PER_SEC);
	timeValue = (num_elements < MAX_CUFFT_ELEMENTS)? timeValue : timeValue*4;
	printf(" %4.2f [ms]\n", timeValue*1000);

    // cleanup memory
	cufftDestroy(plan);
	cudaFree(data);

    cudaThreadExit();
}

void selectDevice(int argc, char** argv) 
{
    int devID;
    cudaDeviceProp deviceProps;

	if( cutCheckCmdLineFlag(argc, (const char**)argv, "device") ) {
	    devID = cutilDeviceInit(argc, argv);
            if (devID < 0) {
               printf("exiting...\n");
               cutilExit(argc, argv);
               exit(0);
            }
	}
	else {
	    devID = cutGetMaxGflopsDeviceId();
	    cudaSetDevice( devID );
	}
		
    // get number of SMs on this GPU
    cutilSafeCall(cudaGetDeviceProperties(&deviceProps, devID));
    printf("CUDA device [%s] has %d Multi-Processors\n", deviceProps.name, deviceProps.multiProcessorCount);

}
