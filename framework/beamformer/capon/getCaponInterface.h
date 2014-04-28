
#ifndef GETCAPONCINTERFACE_H
#define GETCAPONCINTERFACE_H

#include "Complex.h"
#include "MexArray.h"


void getCaponCInterface(ComplexMexArray<double> *z,
                        ComplexMexArray<double> *w,
                        ComplexMexArray<double> *zPow,
                        ComplexMexArray<double> *Xd,
                        float d,
                        int L,
                        int Navg,
//                        int Nx,
//                        int Ny,
//                        int M,
                        ComplexMexArray<double> *V,
                        bool doFBAvg);

#endif
