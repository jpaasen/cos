/**
* Part of the RealTimeCapon (RTC) library. 
*
* Author: Jon Petter Aasen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#pragma once

template <typename C, typename T>
class ICapon
{
public:

	virtual ~ICapon() {};

	/**
	* Method for calculation of Capon weights
   * Return int error code, success = 0.
	**/
	virtual int getCapon(
		C*  &z,				// output amplitude per pixel
		C*  &w,				// output weights per pixel
		C*  &R,				// buffer holding the resulting covariance matrices
		C*  &x,				// buffer holding data vectors
		T   &d,				// diagonal loading factor
		int &L,			   // number of spatial sublengths
		int &Yavg,			// number of samples averaged in time
		int &M,			   // number of data elements
		int &Nx,			   // number of data vectors in azimuth
		int &Nz,			   // number of data vectors in range
		int &Nb			   // dimension of beamspace
		// TODO: add suport for custom beamspace matrix
		) = 0;		
};
