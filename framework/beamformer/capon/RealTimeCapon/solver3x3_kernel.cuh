#ifndef _SOLVER3X3KERNEL_KERNEL_H_
#define _SOLVER3X3KERNEL_KERNEL_H_

#include <cuComplex.h>

int solve3x3(cuComplex* x, const cuComplex* A, const cuComplex* b, const int batch);

#endif
