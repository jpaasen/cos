/**
* Part of the RealTimeCapon (RTC) library. 
*
* Author: Jon Petter Aasen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#pragma once

#include "IBuildR.h"
#include <Complex.h>

class BuildR : IBuildR<Complex<float>, float>
{
public:
   BuildR(void);
   ~BuildR(void);

   int getR(
      Complex<float>* &R_in,		// buffer holding the resulting covariance matrices
      Complex<float>* &x_in,		// buffer holding data vectors
      float &d,			         // diagonal loading factor
      int &L,			            // number of spatial sublengths
      int &Yavg,			         // number of samples averaged in time
      int &M,			            // number of data elements
      int &Nx,			            // number of data vectors in azimuth
      int &Nz,			            // number of data vectors in range
      int &Nb,			            // dimension of beamspace
      // TODO: add suport for custom beamspace matrix
      bool &R_on_gpu,	         // true if you want R to be left on the gpu
      bool &x_on_gpu		         // true if x is already on the gpu
      );	
};
