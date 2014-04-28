/**
* Part of the RealTimeCapon (RTC) library. 
*
* Author: Jon Petter Aasen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/

#pragma once

#include "cuComplex.h"

typedef unsigned int uint;

/**
* Calculates the beamspace transformation of x divided into L long subarrays.
* For each M-L+1 subarray in one M long data vector located in x the function calculates
* Nb spatial beams. If x[i] is M long then x_bs[i] contains is (M-L+1) non-overlapping subarrays of length Nb.
* 
* Limitations: 
*   Nb <= L.
**/
int sliding_beamspace(const cuComplex* x,
                      cuComplex* x_bs,
                      cuComplex* s,  
                      const int M,
                      const int L,
                      const int Nb,
                      const int Nx, 
                      const int Ny);