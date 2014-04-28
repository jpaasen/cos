#ifndef _SOLVER1X1KERNEL_KERNEL_H_
#define _SOLVER1X1KERNEL_KERNEL_H_

#include <cuComplex.h>

int solve1x1(cuComplex* x, const cuComplex* A, const cuComplex* b, const int batch);

#endif
