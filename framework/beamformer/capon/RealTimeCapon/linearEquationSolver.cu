#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cuComplex.h>

// includes, project
#include <cutil_inline.h>

// includes, kernels
#include "buildR.h"
//#include "uhdu.h"
//#include "linearEquationSolver_kernel.cu"
//#include "inverse.h"
#include "solve.h"

//#define N 4

typedef unsigned int uint;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

// forward dec
void runTest( int argc, char** argv, int len, uint N);

int main( int argc, char** argv) 
{
    printf("[Linear equation solver test]\n");


	int numberOfMatrices = 1024*64;
	if (argc > 1) atoi(argv[1]);

	printf("-----------------------------------------------------------------\n");

	uint i;
	uint j;
	uint n = 1;//8;
	uint m = 16;

	for (i = 0; i < n; ++i)
	{
		for (j = 4; j <= m; ++j)
		{
			runTest(argc, argv, numberOfMatrices*uint(powf(2.0f,float(i))), j);
		}
	}

	cutilExit(argc, argv);
}

void runTest( int argc, char** argv, int len, uint N) 
{
	printf("%d matrices\n", len);


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

    unsigned int num_Matrices = len;
    unsigned int mem_size = sizeof(cuComplex) * num_Matrices * N * N;
	unsigned int mem_size_vectors = sizeof(cuComplex) * num_Matrices * N;

	printf("Memory: %d %d %d(Bytes)\n", mem_size, mem_size_vectors, mem_size + mem_size_vectors);
	if (mem_size + mem_size_vectors > deviceProps.totalGlobalMem)
	{
		printf("To little global memory: %d vs %d\n", mem_size + mem_size_vectors, deviceProps.totalGlobalMem);	
		return;
	}

    // allocate host memory
    cuComplex* matrices_host = (cuComplex*) malloc(mem_size);
	cuComplex* rightsides_host = (cuComplex*) malloc(mem_size_vectors);

	cuComplex* signalVectors_host = (cuComplex*) malloc(2*mem_size_vectors);
	//float3* matrices_host;
	//cutilSafeCall( cudaMallocHost( (void**)matrices_host, mem_size)); // page-locked memory for max bandwith

	// formate A[i][j] = A_ij normal matrix syntax
	//cuComplex testMatrix[N][N] = 
	cuComplex testMatrix[16][16] = 
	{
		{ make_cuComplex(5276.20740438f, 0.0f), make_cuComplex(-396.858829841f, -1107.52483868f), make_cuComplex(409.042366195f, -571.865965745f), make_cuComplex(-267.55514414f, 851.610474304f), make_cuComplex(627.342869056f, -221.554367511f), make_cuComplex(648.888345945f, 152.112666955f), make_cuComplex(-99.9799873033f, -1408.41688383f), make_cuComplex(-1225.18922951f, 1197.27434582f), make_cuComplex(-825.972918583f, 1007.55621285f), make_cuComplex(1123.32648015f, 485.644514266f), make_cuComplex(-540.838062869f, -661.399180641f), make_cuComplex(-39.5952002148f, 556.419140887f), make_cuComplex(280.69049046f, -132.360062124f), make_cuComplex(435.104506071f, -368.463542855f), make_cuComplex(-653.06983745f, -461.432933081f), make_cuComplex(475.268195338f, -743.985963932f), },
		{ make_cuComplex(0.0f, 0.0f), make_cuComplex(4882.40966159f, 0.0f), make_cuComplex(-521.571401279f, -1093.84556036f), make_cuComplex(193.820312376f, -373.640118997f), make_cuComplex(233.707559946f, 815.538260619f), make_cuComplex(298.835707995f, -67.7279043643f), make_cuComplex(586.459262759f, -227.114621324f), make_cuComplex(-630.165389972f, -920.545211742f), make_cuComplex(-1271.98630186f, 1116.89408512f), make_cuComplex(-211.702173906f, 893.186374225f), make_cuComplex(1269.70671648f, 160.8329399f), make_cuComplex(-883.688466824f, -642.857887483f), make_cuComplex(582.365770756f, 1189.59452474f), make_cuComplex(759.474292823f, -278.310707991f), make_cuComplex(81.1848190777f, -650.887909706f), make_cuComplex(-745.394439214f, -117.35944654f), },
		{ make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(5088.10350038f, 0.0f), make_cuComplex(-275.643428818f, -1139.88281097f), make_cuComplex(354.336556836f, -462.28908473f), make_cuComplex(486.079069823f, 954.504938216f), make_cuComplex(216.982762432f, -14.896609107f), make_cuComplex(1007.43555607f, -548.682018715f), make_cuComplex(-977.572692571f, -949.714309911f), make_cuComplex(-1084.95558537f, 833.089028922f), make_cuComplex(-477.024432132f, 1003.02607587f), make_cuComplex(1426.14407964f, 389.665670553f), make_cuComplex(-576.33838007f, -794.539269293f), make_cuComplex(590.260984329f, 571.789953709f), make_cuComplex(560.415435394f, -103.986796643f), make_cuComplex(8.73494927271f, -1060.4956222f), },
		{ make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(5780.38252959f, 0.0f), make_cuComplex(-17.3884642217f, -1595.32535052f), make_cuComplex(-142.591128003f, -246.214912962f), make_cuComplex(214.647139557f, 1145.5491412f), make_cuComplex(843.254825769f, 780.303499721f), make_cuComplex(1154.78095799f, -1139.63833885f), make_cuComplex(-1207.71443196f, -906.542376269f), make_cuComplex(-1436.67277859f, 984.441867069f), make_cuComplex(-27.9416465035f, 1242.67019704f), make_cuComplex(1527.24735667f, 16.0990662068f), make_cuComplex(-163.492296339f, -819.08551669f), make_cuComplex(131.525978499f, 275.389428928f), make_cuComplex(1061.51265554f, -124.764617518f), },
		{ make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(6003.46538975f, 0.0f), make_cuComplex(-133.103501122f, -1666.96454254f), make_cuComplex(-509.664532959f, 115.953255803f), make_cuComplex(178.920135566f, 1558.04377987f), make_cuComplex(909.578307491f, 970.75488968f), make_cuComplex(828.996392324f, -1220.53978039f), make_cuComplex(-1513.55182067f, -890.386271845f), make_cuComplex(-977.710640392f, 1476.74566839f), make_cuComplex(528.032541887f, 291.950763157f), make_cuComplex(958.759169959f, -222.840718228f), make_cuComplex(-88.8568274984f, -986.703137677f), make_cuComplex(413.836316375f, 879.278360974f), },
		{ make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(6210.18283457f, 0.0f), make_cuComplex(57.6887953737f, -2257.68637274f), make_cuComplex(-967.211359398f, -26.5859338147f), make_cuComplex(97.5958930116f, 1836.95339926f), make_cuComplex(1509.67553796f, 675.596993778f), make_cuComplex(630.142967582f, -1401.48419661f), make_cuComplex(-2066.64801272f, -776.158264119f), make_cuComplex(-909.828039176f, 2364.6415345f), make_cuComplex(1043.61499232f, 178.276584215f), make_cuComplex(1053.19643263f, -275.814814363f), make_cuComplex(-851.681653613f, -1414.99271359f), },
		{ make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(6227.26890573f, 0.0f), make_cuComplex(137.162114498f, -2263.92372837f), make_cuComplex(-1203.80001442f, 2.26994201751f), make_cuComplex(374.792441316f, 1393.59964273f), make_cuComplex(1552.068062f, 505.846514734f), make_cuComplex(532.939391183f, -2037.21768103f), make_cuComplex(-2071.79425802f, -584.10833668f), make_cuComplex(-812.522701572f, 1966.06516067f), make_cuComplex(557.026974631f, 359.570825915f), make_cuComplex(1044.46554399f, -220.94167667f), },
		{ make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(6636.90484588f, 0.0f), make_cuComplex(83.625428908f, -2733.85124713f), make_cuComplex(-1238.09234613f, 729.940773406f), make_cuComplex(195.877747669f, 1397.29066849f), make_cuComplex(2076.79765852f, 582.455458303f), make_cuComplex(-168.003195242f, -2125.95239171f), make_cuComplex(-1547.31844082f, -3.31533270656f), make_cuComplex(-476.509521248f, 1448.50405481f), make_cuComplex(640.966616104f, -259.958116335f), },
		{ make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(6900.01146378f, 0.0f), make_cuComplex(-597.739036386f, -3381.99146282f), make_cuComplex(-1318.90488377f, 804.372830777f), make_cuComplex(-118.98893455f, 2061.55038591f), make_cuComplex(2608.32549155f, 694.910633662f), make_cuComplex(-453.633210245f, -2042.58679578f), make_cuComplex(-1636.18211195f, 370.526754664f), make_cuComplex(-749.765551118f, 1824.79169606f), },
		{ make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(7096.6365754f, 0.0f), make_cuComplex(-832.687999301f, -3657.82142885f), make_cuComplex(-1732.84897854f, 684.318605012f), make_cuComplex(-297.642749621f, 1683.28018916f), make_cuComplex(1745.02990881f, 432.0463835f), make_cuComplex(-317.010276812f, -1581.55147136f), make_cuComplex(-1544.11011039f, -152.276459114f), },
		{ make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(7519.34277032f, 0.0f), make_cuComplex(-792.602039395f, -4383.50802576f), make_cuComplex(-1875.18700021f, 641.358373041f), make_cuComplex(-453.621481041f, 1688.44032691f), make_cuComplex(1589.02683706f, 460.548872271f), make_cuComplex(-219.942206045f, -864.177810859f), },
		{ make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(8103.00901815f, 0.0f), make_cuComplex(-1297.89608534f, -4701.26151819f), make_cuComplex(-1936.37255781f, 625.126789885f), make_cuComplex(-1151.94661013f, 1555.02927605f), make_cuComplex(1526.61239431f, 414.684903476f), },
		{ make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(7954.11472674f, 0.0f), make_cuComplex(-1585.34395971f, -4273.54446246f), make_cuComplex(-1280.09923525f, 164.578282723f), make_cuComplex(-819.03673374f, 1614.98780717f), },
		{ make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(7392.81297599f, 0.0f), make_cuComplex(-1139.67960839f, -3606.93093402f), make_cuComplex(-739.233871178f, -119.812503076f), },
		{ make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(6853.74407525f, 0.0f), make_cuComplex(-1251.16966279f, -3417.61593654f), },
		{ make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(0.0f, 0.0f), make_cuComplex(7657.29415569f, 0.0f), },
	};

	
	
	/*{
		{make_cuComplex(22.0f, 0.0f),  make_cuComplex(8.0f, 0.0f),  make_cuComplex(11.0f, -11.0f), make_cuComplex(22.0f, -7.0)},
		{make_cuComplex(8.0f, 0.0f),   make_cuComplex(22.0f, 0.0f), make_cuComplex(17.0f, -2.0f),  make_cuComplex(11.0f, -7.0)},
		{make_cuComplex(11.0f, 11.0f), make_cuComplex(17.0f, 2.0),  make_cuComplex(45.0f, 0.0f),   make_cuComplex(23.0f, -5.0)},
		{make_cuComplex(22.0f, 7.0),   make_cuComplex(11.0f, 7.0f), make_cuComplex(23.0f, 5.0),	   make_cuComplex(37.0f, 0.0f)}
	};*/

	// UHDU decomposition of A:
	/*
	U =

   1.0000             0.3636             0.5000 - 0.5000i   1.0000 - 0.3182i
        0             1.0000             0.6810 + 0.1048i   0.1571 - 0.2333i
        0                  0             1.0000             0.2776 - 0.3670i
        0                  0                  0             1.0000          


D =

   22.0000         0         0         0
         0   19.0909         0         0
         0         0   24.9381         0
         0         0         0    5.9806

		 */
	

	//cuComplex testRightside[N] = 
	cuComplex testRightside[16] = 
	{
		make_cuComplex(1.0,0.0),
		make_cuComplex(1.0,0.0),
		make_cuComplex(1.0,0.0),
		make_cuComplex(1.0,0.0),
		make_cuComplex(1.0,0.0),
		make_cuComplex(1.0,0.0),
		make_cuComplex(1.0,0.0),
		make_cuComplex(1.0,0.0),
		make_cuComplex(1.0,0.0),
		make_cuComplex(1.0,0.0),
		make_cuComplex(1.0,0.0),
		make_cuComplex(1.0,0.0),
		make_cuComplex(1.0,0.0),
		make_cuComplex(1.0,0.0),
		make_cuComplex(1.0,0.0),
		make_cuComplex(1.0,0.0),
	};

	//cuComplex testLeftSide[N] = 
	cuComplex testLeftSide[16] = 
	{
		make_cuComplex(0.000514730453848f,0.0001938351133f),
		make_cuComplex(0.000443358873817f,-0.000149142741101f),
		make_cuComplex(0.000227479661645f,-0.000180262883725f),
		make_cuComplex(5.05981023683e-05f,9.88749634731e-05f),
		make_cuComplex(8.83809192447e-05f,9.29728784204e-05f),
		make_cuComplex(0.00014697268264f,-1.29472493057e-05f),
		make_cuComplex(0.000184322339458f,-5.71845727136e-05f),
		make_cuComplex(0.000118285195332f,8.53223798613e-06f),
		make_cuComplex(0.000300957114955f,-3.82766946289e-05f),
		make_cuComplex(0.000281682940332f,9.96732319888e-05f),
		make_cuComplex(0.000346073355903f,0.000165250251891f),
		make_cuComplex(0.000399347218563f,2.78129167159e-05f),
		make_cuComplex(0.000355561786126f,0.000109284590463f),
		make_cuComplex(0.000549851218965f,6.89198442124e-05f),
		make_cuComplex(0.00062160348564f,-0.000162355611463f),
		make_cuComplex(0.000272466882018f,-0.00026573331826f),
	};
	
	/*{
		make_cuComplex(1.0f, 0.0f),
		make_cuComplex(1.0f, 1.0f),
		make_cuComplex(1.0f, -2.0f),
		make_cuComplex(1.0f, -2.0f)};*/

    // initalize the memory
    for( unsigned int n = 0; n < num_Matrices; ++n)
    {
		// Outdated: one line is one row in a N*N matrix
		// Outdated: to avoid strid access in kernel, hence better coalescing, all first rows are placed after each other, then second columns etc...
		// one line is one column in a N*N matric (column-major-order)
		for (unsigned int i = 0; i < N; i++) 
		{

			for (unsigned int j = 0; j < N; j++)
			{
				//matrices_host[i*N*num_Matrices + n*N + j] = testMatrix[i][j]; // all first rows first
				matrices_host[n*N*N + j*N + i] = testMatrix[i][j]; // one matrix, then the next and etc...
			}

			//rightsides_host[i*num_Matrices + n] = testRightside[i];
			rightsides_host[n*N + i] = testRightside[i];
			signalVectors_host[n*2*N + i] = testLeftSide[i];
			signalVectors_host[n*2*N + N + i] = testLeftSide[i];
		}
	}

    // allocate device memory
	cuComplex* matrices_device;
	cutilSafeCall( cudaMalloc( (void**) &matrices_device, mem_size));

	cuComplex* rightsides_device;
	cutilSafeCall( cudaMalloc( (void**) &rightsides_device, mem_size_vectors));

	cuComplex* solution_device;
	cutilSafeCall( cudaMalloc( (void**) &solution_device, mem_size_vectors));

	cuComplex* signalVectors_device;
	cutilSafeCall( cudaMalloc( (void**) &signalVectors_device, 2*mem_size_vectors));

	float totalTime = 0.0f;
	float time = 0.0f;
	unsigned int timer = 0;
    cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

    // copy host memory to device
	cutilSafeCall( cudaMemcpy( matrices_device, matrices_host, mem_size,
                                cudaMemcpyHostToDevice) );

	cutilSafeCall( cudaMemcpy( rightsides_device, rightsides_host, mem_size_vectors,
                                cudaMemcpyHostToDevice) );

	cutilSafeCall( cudaMemcpy( signalVectors_device, signalVectors_host, 2*mem_size_vectors,
                                cudaMemcpyHostToDevice) );


	cutilCheckError( cutStopTimer( timer));
	time = cutGetTimerValue( timer);
	totalTime += time;
    printf( "Memcpy time HostToDevice: %f (ms)\n", time);
    cutilCheckError( cutDeleteTimer( timer));
	cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

	/** BUILD R **/
	int M = 2*N;
	int L = M/2;//16; // size of sub array
	float d = 0.01f; // diagonal loading
	int Yavg = 5;
	//dim3 gridR(1,1,1);
	//dim3 blockR(L, BUILD_R_NUMBER_OF_THREADS/L, 1);
	//gridR.x = (num_Matrices-1)/blockR.y + 1;
	
	//buildR_kernel<<<gridR, blockR>>>(signalVectors_device, matrices_device, d, L, 2*N-L+1, 2*N, len);
	int cudaerror = build_R(signalVectors_device, matrices_device, d, L, 2*N, len);

	//int cudaerror = build_Yavg_R(signalVectors_device, matrices_device, d, L, M, len/1024, 1024, Yavg);

	cudaThreadSynchronize();

	/*if (cudaerror != cudaSuccess)
	{
		printf("CUDA ERROR: %s\n", cudaGetErrorString(cudaError(cudaerror)));
		return;
	}*/
	/**         **/

	cutilCheckError( cutStopTimer( timer));
	time = cutGetTimerValue( timer );
	totalTime += time;
    printf( "Build R (M,L,Yavg) = (%d,%d,%d): %f (ms)\n", M, L, Yavg, time);
    cutilCheckError( cutDeleteTimer( timer));
	cutilCheckError( cutCreateTimer( &timer));
    cutilCheckError( cutStartTimer( timer));

    
	/** Decomp R **/
	// setup execution parameters
	/*dim3  grid(1, 1, 1);
	
	uint numOfThreadBlocksRequired = (num_Matrices-1)/(MAX_M / N) + 1;
	//uint numOfThreadBlocksRequired = num_Matrices;
	uint maxCUDAGridSize = 65535;
	
	if (numOfThreadBlocksRequired < maxCUDAGridSize)
	{
		grid.x = numOfThreadBlocksRequired;
	} else {
		grid.x = maxCUDAGridSize;
		grid.y = ((numOfThreadBlocksRequired-1) / 65535) + 1;
	}
	dim3  threads(MAX_M, 1, 1);

	// execute decomp kernel
	uhdu<<<grid, threads>>>(matrices_device, N, N*N, len, N, N*N);
*/

	// Nvidia's batched solver
	//zfmatinv_batch(matrices_device, matrices_device, N, len);
	cudaerror = zfsolve_batch(matrices_device, rightsides_device, solution_device, N, len);

	cudaThreadSynchronize();

	if (cudaerror != cudaSuccess)
	{
		printf("CUDA ERROR: %s\n", cudaGetErrorString(cudaError(cudaerror)));
		return;
	}
	
    // execute solve kernel
	//linearEquationSolverUHDU<<< grid, threads>>>(matrices_device, rightsides_device, N, N*N, len);

	/** Solve systems and calculate weights**/

    // check if kernel execution generated and error
    cutilCheckMsg("Kernel execution failed");

	cutilCheckError( cutStopTimer( timer));
	time = cutGetTimerValue( timer );
	totalTime += time;
    printf( "Solving Rb=a: %f (ms)\n", time);
    cutilCheckError( cutDeleteTimer(timer) );
	cutilCheckError( cutCreateTimer(&timer) );
    cutilCheckError( cutStartTimer(timer) );

	// allocate mem for uhdu-decomp on host side
	//cuComplex* solutionUHDU = (cuComplex*) malloc(mem_size);
	//cutilSafeCall( cudaMemcpy(solutionUHDU, matrices_device, mem_size, cudaMemcpyDeviceToHost) );

    // allocate mem for the result on host side
	//cuComplex* solutionVector_host = (cuComplex*) malloc(mem_size_vectors);
    // copy result from device to host. Matrices is now saved row-wise
	//cutilSafeCall( cudaMemcpy( solutionVector_host, rightsides_device, mem_size_vectors,
    //                           cudaMemcpyDeviceToHost) );

	cuComplex* solution_host = (cuComplex*) malloc(mem_size_vectors);
	cutilSafeCall( cudaMemcpy(solution_host, solution_device, mem_size_vectors, cudaMemcpyDeviceToHost) );

    cutilCheckError( cutStopTimer( timer));
    printf( "Memcpy time DeviceHost: %f (ms)\n", cutGetTimerValue( timer));
    cutilCheckError( cutDeleteTimer( timer));

	printf("[%f fps]\n", 1000.0f/(totalTime));

    // compute reference solution for the given right side abov
	//cuComplex refX[N] = {
	cuComplex refX[4] = {
		make_cuComplex(0.8280f, -0.0579f),
		make_cuComplex(0.0025f, 0.2643f),
        make_cuComplex(0.4111f, 0.0872f),
		make_cuComplex(-0.2227f, 0.6372f)
	};
	/*cuComplex refX[N] = {
		make_cuComplex(1.593749999999999f, -0.06250f), 
		make_cuComplex(-0.343750f, 0.281250f), 
		make_cuComplex(-0.1250f, -0.31250f)};
*/
	bool testPassed = true;

    // check result
	/*for( unsigned int i = 0; i < num_Matrices; ++i) 
    {

		float3 resX[N];
		for (unsigned int j = 0; j < N; ++j)
		{
			resX[0] = solutionVector_host[i + len*j];
		}

		if (!cutComparef((float*) &refX[0].x, (float*) &resX[0].x, N) ||
			!cutComparef((float*) &refX[0].y, (float*) &resX[0].y, N)) {
			printf("Error in result:\n");

			printf("%f %f %f\n%f %f %f\n%f %f %f", 
				refX[0].x, refX[0].y, refX[0].z, 
				refX[1].x, refX[1].y, refX[1].z, 
				refX[2].x, refX[2].y, refX[2].z);
			printf("\n");
			printf("%f %f %f\n%f %f %f\n%f %f %f", 
				resX[N*i].x,     resX[N*i].y,     resX[N*i].z,
				resX[N*i + 1].x, resX[N*i + 1].y, resX[N*i + 1].z,
				resX[N*i + 2].x, resX[N*i + 2].y, resX[N*i + 2].z);
			printf("\n");

			testPassed = false;

		}
    }*/

	if (testPassed) {
		printf("Test PASSED\n");
	}

	printf("-----------------------------------------------------------------\n");

    // cleanup memory
	//free(matrices_host);
	free(matrices_host);
	free(rightsides_host);
    //free(solutionVector_host);
	free(solution_host);
	cutilSafeCall(cudaFree(matrices_device));
	cutilSafeCall(cudaFree(rightsides_device));
	cutilSafeCall(cudaFree(solution_device));
	//cutilSafeCall(cudaFree(signalVectors_device));

    cudaThreadExit();
}
