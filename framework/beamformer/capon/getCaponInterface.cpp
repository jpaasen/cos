
#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdio.h>

//#include "Complex.h"
#include "MexArray.h"

#include <iostream>
#include <stdlib.h>
#include <time.h>

// #ifdef __cplusplus
// extern "C" {
// #include <getCaponC.h>
// }
// #else
#include "getCaponC.h"
// #endif

//#define VERBOSE 1


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

int main() {

   int Nx   = 2;
   int Ny   = 2;
   int M    = 4;
   int L    = 2;
   int d    = 0.02;
   int Navg = 0;
   bool doFBAvg = false;

   size_t w_shape[3]  = {Nx,Ny,L};
   size_t Xd_shape[3] = {Nx,Ny,M};
   size_t z_shape[2]  = {Nx,Ny};
   size_t V_shape[2]  = {1,1};

   mxClassID mx_class = mxDOUBLE_CLASS;

   std::cout << "Am I even alive???\n";

   ComplexMexArray<double> *z    = new ComplexMexArray<double>(2, z_shape, mx_class);
   ComplexMexArray<double> *w    = new ComplexMexArray<double>(3, w_shape, mx_class);
   ComplexMexArray<double> *zPow = new ComplexMexArray<double>(2, z_shape, mx_class);
   ComplexMexArray<double> *Xd   = new ComplexMexArray<double>(3, Xd_shape, mx_class);
   ComplexMexArray<double> *V    = new ComplexMexArray<double>(2, V_shape, mx_class);

   srand ( time(NULL) );
   for( int x=0; x<Nx*Ny*M; ++x ){
      Xd->data[x].re = (rand()%100)/100.0;
      Xd->data[x].im = (rand()%100)/100.0;
      std::cout << x << " ";
   }

   std::cout << "Running interface\n";
   getCaponCInterface(z,w,zPow,Xd,d,L,Navg,V,doFBAvg);

   std::cout << "Result\n";
   for( int x=0; x<Nx; ++x ){
      for( int y=0; y<Ny; ++y ){
         std::cout << "(" << z->data[Ny*(y/Ny)+y].re << "+j" << z->data[Ny*(y/Ny)+y].im << ") ";
      }
      std::cout << "\n";
   }

   return 0;
}


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
                        bool doFBAvg)
{
   std::cout << "Entering CaponInterface.cpp" << std::endl;

   Complex<double> *Xd_ptr, *V_ptr, *w_ptr, *z_ptr, *zPow_ptr;

   size_t Nx = Xd->shape[0];
   size_t Ny = Xd->shape[1];
   size_t M  = Xd->shape[2];

   size_t w_shape[3];
   w_shape[0] = Nx;
   w_shape[1] = Ny;
   w_shape[2] = L;
   
   size_t Xd_shape[3];
   Xd_shape[0] = Nx;
   Xd_shape[1] = Ny;
   Xd_shape[2] = M;
   
   size_t z_shape[2];
   z_shape[0] = Nx;
   z_shape[1] = Ny;
   
   size_t V_shape[2];
   V_shape[0] = 1;
   V_shape[1] = 1;

   std::cout << "Initializing\n";
   Py_Initialize();
      
    // THIS is the solution!!!
   import_array();
//    PyArrayObject *Xd_py = (PyArrayObject*)PyArray_SimpleNewFromData(Xd->dims, (npy_intp*)Xd->shape, NPY_COMPLEX128, (void*)Xd->data);
// 
//    PyArrayObject *V_py = (PyArrayObject*)PyArray_SimpleNewFromData(V->dims, (npy_intp*)V->shape, NPY_COMPLEX128, (void*)V->data);

   PyArrayObject *Xd_py  = (PyArrayObject*)PyArray_SimpleNew(3, (npy_intp*)Xd_shape, NPY_COMPLEX128);
   PyArrayObject *V_py   = (PyArrayObject*)PyArray_SimpleNew(2, (npy_intp*)V_shape, NPY_COMPLEX128);
   Py_INCREF(Xd_py);
   Py_INCREF(V_py);

   std::cout << "Xd in getCaponInterface.cpp\n";
   Xd_ptr = (Complex<double> *)PyArray_DATA(Xd_py);
   for(size_t i=0; i<Xd->size; ++i) {
      Xd_ptr[i] = Xd->data[i];
      std::cout << "(" << Xd_ptr[i].re << "+j" << Xd_ptr[i].im << ") ";
   }
   std::cout << "\n";

   V_ptr = (Complex<double> *)PyArray_DATA(V_py);
   for(size_t i=0; i<V->size; ++i)
      V_ptr[i] = V->data[i];
   
//    Xd_py->data = (char*)Xd->data;
//    V_py->data = (char*)V->data;

   int zz_shape[2];
   zz_shape[0] = Nx;
   zz_shape[1] = Ny;

   PyArrayObject *w_py    = (PyArrayObject*)PyArray_SimpleNew(3, (npy_intp*)w_shape, NPY_COMPLEX128);
   PyArrayObject *z_py    = (PyArrayObject*)PyArray_SimpleNew(2, (npy_intp*)z_shape, NPY_COMPLEX128);
   PyArrayObject *zPow_py = (PyArrayObject*)PyArray_SimpleNew(2, (npy_intp*)z_shape, NPY_COMPLEX128);
   Py_INCREF(w_py);
   Py_INCREF(z_py);
   Py_INCREF(zPow_py);

   
   std::cout << "Running getCapon\n";
   initgetCaponC();
   getCapon(z_py, w_py, zPow_py, Xd_py, d, L, Navg, V_py, false, false);
   
   w    = new ComplexMexArray<double>(3, w_shape, Xd->mx_class);
   z    = new ComplexMexArray<double>(2, z_shape, Xd->mx_class);
   zPow = new ComplexMexArray<double>(2, z_shape, Xd->mx_class);
   
   w_ptr    = (Complex<double>*)PyArray_DATA(w_py);
   z_ptr    = (Complex<double>*)PyArray_DATA(z_py);
   zPow_ptr = (Complex<double>*)PyArray_DATA(zPow_py);

   for(size_t i=0; i<w->size; ++i) {
      w->data[i] = (Complex<double>)w_ptr[i];
   }
   for(size_t i=0; i<z->size; ++i) {
      z->data[i]    = (Complex<double>)z_ptr[i];
      zPow->data[i] = (Complex<double>)zPow_ptr[i];
   }
//   w->data    = (Complex<double>*)PyArray_DATA(w_py);
//   z->data    = (Complex<double>*)PyArray_DATA(z_py);
//   zPow->data = (Complex<double>*)PyArray_DATA(zPow_py);
   
   std::cout << "z in getCaponInterface.cpp after call to getCapon\n";
   for(size_t i=0; i<z->size; ++i) {
      std::cout << "(" << z->data[i].re << "+j" << z->data[i].im << ") ";
   }
   
   delete Xd_ptr, delete V_ptr, delete z_ptr, delete w_ptr, zPow_ptr, delete Xd_py, delete V_py, delete w_py, delete z_py, delete zPow_py;

   if( Py_IsInitialized() ){
      Py_Finalize();
   }

}


