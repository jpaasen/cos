/**
* Part of the RealTimeCapon (RTC) library. 
*
* Author: Jon Petter Aasen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#pragma once

#ifndef MEMORY_ONLY
#include "cuComplex.h"
#else
#include "cuComplex_memonly.h"
#endif

typedef unsigned int uint;

int build_R(const cuComplex* x, 
            cuComplex* R, 
            const float d, 
            const uint L, 
            const uint Yavg, 
            const uint M, 
            const uint Nx, 
            const uint Ny);

int build_R_full(const cuComplex* x, 
                 cuComplex* R, 
                 const float d, 
                 const int L, 
                 const int Yavg, 
                 const int M, 
                 const int Nx, 
                 const int Ny, 
                 const int subarrayStrid);
