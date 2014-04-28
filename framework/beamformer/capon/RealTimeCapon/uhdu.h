#ifndef _UHDU_HEADER_
#define _UHDU_HEADER_

#include "cuComplex.h"
#include "cudaConfig.h"

typedef unsigned int uint;

#define UHDU_NUMBER_OF_THREADS 256

int uhdu(cuComplex* A, const uint m, const uint n);

#endif