/**
* Part of the RealTimeCapon (RTC) library. 
*
* Author: Jon Petter Aasen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#pragma once

template <typename C, typename T>
class IBuildR
{
public:

   virtual ~IBuildR() {};

   /**
   * Method for estimating multiple sample covariance matrices R 
   * from a set of data vectors x.
   *
   * R is returned in column-major order.
   *
   * d: is a diagonal factor added to all R's. It can be either constant
   * or relative to R depending on the implementation.
   *	
   * L: each vector x can be divided into (overlapping-all-but-one) sublenghts of length L. 
   * The resulting R is L*L, and is the average of all sublengths (K = M-L+1).
   *
   * Yavg: R is averaged over 2*Yavg+1 data vectors in range.
   *
   * M: is the number of elements in each data vector x.
   *
   * Nx and Nz: The method supports a 2D grid (Nx, Nz) of data vectors.
   *
   * Nb: Dimension of beamspace. Each sublength can be reduced by
   * a beamspace transformation. The size of each R will be Nb*Nb. 
   * Nb = 0 will disable beamspace. The sum of x transformed will be returned in x,
   * now occupying Nb*Nx*Nz memory.  
   *
   * R_on_gpu: Set this to true if you know the implementation is using the GPU,
   *			and you want R to remain in device memory. Input R should then be a buffer in gpu-memory.
   * x_on_gpu: Set this to true if x is already in device memory.
   * If set to false, the method will allocate the required memory if the computation does 
   * not take place on the host (CPU) side.
   *
   * Template argument C can be complex or real. T should have the same precision as real(C).
   *
   * R and x is not freed by this method.
   *
   * Return value is an error code (for GPU/CUDA implementations this is a cudaError).
   **/
   virtual int getR(
      C* &R_in,			// buffer holding the resulting covariance matrices
      C* &x_in,		// buffer holding data vectors
      T &d,			// diagonal loading factor
      int &L,			// number of spatial sublengths
      int &Yavg,			// number of samples averaged in time
      int &M,			// number of data elements
      int &Nx,			// number of data vectors in azimuth
      int &Nz,			// number of data vectors in range
      int &Nb,			// dimension of beamspace
      // TODO: add support for custom beamspace matrix
      bool &R_on_gpu,	// true if you want R to be left on the gpu
      bool &x_on_gpu		// true if x is already on the gpu
      ) = 0;	
};
