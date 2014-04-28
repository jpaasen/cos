/**
* Part of the RealTimeCapon (RTC) library. 
*
* Author: Jon Petter Aasen
* E-mail: jon.p.asen@ntnu.no - jp.aasen@gmail.com
**/
#pragma once

#include <cuComplex.h>

#include "ISolver.h"
#include "nvidia_solve.h"
#include <Complex.h> // from framework lib



class Solver : ISolver<Complex<float> >
{
public:
   
   enum SolverType {
      NVIDIA, // Gauss Jordan solver from Nvidia (CUDA). Supports dimensions from 2 to 72.
      DIRECT  // Use direct solvers for small problems if available, otherwise use NVIDIA
   } solverType;

//   static SolverType solverType;

   Solver(SolverType type = Solver::DIRECT);//NVIDIA);
   ~Solver(void);

   int solve(
      Complex<float>* &x_in,        // buffer holding the solutions
      Complex<float>* &A_in,  // buffer holding matrices
      Complex<float>* &b_in,  // buffer holding the left sides
      int &N,              // size of each linear system
      int &batch,          // number of linear systems
      bool &x_on_gpu,      // true if x should remain on the gpu
      bool &A_on_gpu,      // true if R is already on the gpu
      bool &b_on_gpu       // true if b is already on the gpu
      );

//private:
//   static SolverType solverType;
};
//

//Solver::solverType = (int)Solver::SolverType::DIRECT;
