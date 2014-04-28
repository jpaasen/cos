/**
* Part of the RealTimeCapon (RTC) library. 
*
* Author: Jon Petter Aasen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#ifndef _CAPON_KERNEL_
#define _CAPON_KERNEL_

#include "cuComplex.h"

typedef unsigned int uint;

/** 
* Creates NL scalar values on the GPU.
* If beamspace is true, every L'th value equals value (starting from index 0), the rest is set to zero.
**/
int values(cuComplex* b, float value, int NL, int L, bool beamspace);

/** 
   Creates Npx Butler matrices of size Nb times L on the GPU
   Max Nb is 512.
   Nb is the number of symmetric beams around zero.
   For even number of beams, the left-over-beam is placed on the positiv angular side. 
**/
int butler_matrix(cuComplex* B, int Nb, int L, int Npx);

/**
* Calculates w = Ria/a^HRia, and applies the weights on x -> z = w*x
**/
int getCaponOutput(cuComplex*       z_gpu,   // Pixel amplitude (output)
                   cuComplex*       w_gpu,   // Weights used (output)
                   const cuComplex* x_gpu,   // Data vector
                   const cuComplex* Ria_gpu, // R^(-1)a
                   const int        M,       // Number of channels
                   const int        L,       // Subarray size
                   const int			Yavg,		// Time averaging size (In total 2*Yavg+1 samples)
                   const int			Ny,		// Number of samples in one range line
                   const int        N);      // Number of pixels

/**
* Calculates w, and applies the weights on pre-computed subarray sums.
**/
int getCaponOutputFromSubarraySums(cuComplex*       z_gpu,      // Pixel amplitude (output)
                                   cuComplex*       w_gpu,      // Weights used (output)
                                   const cuComplex* s_gpu,      // Subarray sums vector
                                   const cuComplex* Ria_gpu,    // R^(-1)a
                                   const int        oL,         // Original subarray size (if beamspace has been applied oL is usually larger than L)
                                   const int        L,          // Subarray size
                                   const int        Yavg,       // Temporal averaging
                                   const int        Ny,         // Number of pixels in range
                                   const int        N,          // Number of pixels
                                   const bool       beamspace); // If true, capon weights are normalized with the first element of Ria only, not the sum as in element space

#endif
