/*
 * Copyright (c) 2011, NVIDIA Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 *
 *   Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimer.
 *
 *   Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 *   Neither the name of NVIDIA Corporation nor the names of its contributors
 *   may be used to endorse or promote products derived from this software 
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#if !defined(SOLVE_H_)
#define SOLVE_H_
#include "cuComplex.h"

#ifdef __cplusplus
extern "C" {
#endif

/* dsolve_batch() solves one or many double-precision systems of linear 
   equations, each with a single right-hand side. Partial pivoting is 
   employed by the solver algorithm for increased numerical stability, 
   but no check for singularity is performed.

   A     pointer to an array of double-precision matrices, each stored in 
         column-major order
   b     pointer to an array of double-complex column vectors each representing
         right-hand sides of each corresponding matrix in the array of matrices
         pointed to by A
   x     pointer to an array of double-precision column vectors representing
         the solution vector A x = b of the corresponding pair of matrix and 
         right-hand side in the arrays pointed to by A and b
   n     number of simultaneous equations in each system. n must be greater 
         than, or equal to, 2. On sm_13 GPUs, n must be less than, or equal 
         to, 44. On sm_2x GPUs, n must be less than, or equal to, 76..
   batch the number of systems to be solved. It must be greater than zero.

   Returns:
  
    0    operation completed successfully
   -1    n is out of bounds, or batch is out of bounds
   -2    a CUDA error occured
*/  
int dsolve_batch (double *A, double *b, double *x, int n, int batch);

/* zsolve_batch() solves one or many double-complex systems of linear 
   equations, each with a single right-hand side. Partial pivoting is 
   employed by the solver algorithm for increased numerical stability,
   but no check for singularity is performed.

   A     pointer to an array of double-complex matrices, each stored in column-
         major order
   b     pointer to an array of double-complex column vectors each representing
         right-hand sides of each corresponding matrix in the array of matrices
         pointed to by A
   x     pointer to an array of double-precision column vectors representing
         the solution vector A x = b of the corresponding pair of matrix and 
         right-hand side in the arrays pointed to by A and b
   n     number of simultaenous equations in each system. n must be greater 
         than, or equal to, 2. On sm_13 GPUs, n must be less than, or equal to,
         53. On sm_2x GPUs, n must be less than, or equal to, 31.
   batch the number of systems to be solved. It must be greater than zero.

   Returns:
   
    0    operation completed successfully
   -1    n is out of bounds, batch is out of bounds
   -2    a CUDA error occured
*/  
int zsolve_batch (cuDoubleComplex *A, cuDoubleComplex *b, cuDoubleComplex *x, 
                  int n, int batch);


/* dmatinv() inverts a square, non-singular matrix of double-precision 
   elements. Partial pivoting is employed in the inversion process for 
   increased numerical stability.

   A     pointer to the matrix of double-precision elements to be inverted, 
         stored in column-major order
   Ainv  pointer to the matrix of double-precision elements which receives
         the inverse of the matrix pointed to by A, stored in column-major	
         order
   n     number of rows and columns of the matrix. n must be greater than,
         or equal to, 2. On sm_13 GPUs, n must be less than, or equal to, 44.
         On sm_2x GPUs n must be less than, or equal to, 76.

   Returns:

    0    operation completed successfully
   -1    n is out of bounds
   -2    a CUDA error occured
*/
int dmatinv (double *A, double *Ainv, int n);

/* zmatinv() inverts a square, non-singular matrix of double-complex elements. 
   Partial pivoting is employed in the inversion process for increased 
   numerical stability.

   A     pointer to the matrix of double-complex elements to be inverted, 
         stored in column-major order
   Ainv  pointer to the matrix of double-complex elements which receives
         the inverse of the matrix pointed to by A, stored in column-major 
         order.
   n     number of rows and columns of the matrix. n must be greater than,
         or equal to, 2. On sm_13 GPUs, n must be less than, or equal to, 31.
         On sm_2x GPUs n must be less than, or equal to, 53.

   Returns:

    0    operation completed successfully
   -1    n is out of bounds
   -2    a CUDA error occured
*/
int zmatinv (cuDoubleComplex *A, cuDoubleComplex *Ainv, int n);

// Single precision versions
int fsolve_batch (float *A, float *b, float *x, int n, int batch);
int zfsolve_batch (cuComplex *A, cuComplex *b, cuComplex *x, int n, int batch);

int fmatinv (float *A, float *Ainv, int n);
int zfmatinv (cuComplex *A, cuComplex *Ainv, int n);

#ifdef __cplusplus
}
#endif

#endif /* SOLVE_H_ */
