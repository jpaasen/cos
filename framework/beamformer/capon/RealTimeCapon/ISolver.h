/**
* Part of the RealTimeCapon (RTC) library. 
*
* Author: Jon Petter Aasen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#pragma once

template <typename C>
class ISolver
{
public:

   virtual ~ISolver() {};

   /**
   * Method for solving N sets of linear equations Ax = b -> x = A^{-1}b.
   * A must be in row-major order.
   **/
   virtual int solve(
      C* &x,		      // buffer holding the solutions
      C* &A,			   // buffer holding matrices
      C* &b,			   // buffer holding the right sides
      int &N,			   // size of each linear system
      int &batch,       // number of linear systems
      bool &x_on_gpu,   // true if x should remain on the gpu
      bool &A_on_gpu,   // true if R is already on the gpu
      bool &b_on_gpu	   // true if b is already on the gpu
      ) = 0;	
};
