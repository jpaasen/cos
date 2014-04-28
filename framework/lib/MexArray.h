#ifndef MEXARRAY_H
#define MEXARRAY_H

#include <stdio.h>
#include "Complex.h"
#include "mex.h"

//#define VERBOSE 1

using namespace std;

                                                /*************************************************/
  ///////////////////                           /* Basic class that defines the basic            */
 /// class Array ///                            /* properties of a matrix, and contains operator */
///////////////////                             /* overloads. Types are defined in Enum Types.   */
                                                /*************************************************/

class MexArray
{ 
public:
   size_t        dims;    // Number of dimensions
   size_t*       shape;   // Size of each of the dimensions
   size_t        size;    // Total size
   bool          complex; // Complex data?

   mxClassID mx_class; // Data class (mxSINGLE, mxDOUBLE, mxUINT32_CLASS, mxINT32_CLASS)


   MexArray(size_t dims_in, const size_t *shape_in, mxClassID class_in, bool complex_in)
   /**********************************************************************************************/
   /* Construct MexArray based on parameters relayed through Real/ComplexMexArray constructors   */
   /**********************************************************************************************/
   {
//      mwSize mw_shape_in = *shape_in;

      dims     = (mwSize)dims_in;
      complex  = complex_in;
      mx_class = class_in;
      shape    = new size_t[dims];
      
      size = 1;
      for( size_t i=0; i<dims; ++i) {
         shape[i] = shape_in[i];
         size    *= shape[i];
      }

   }

   MexArray(const mxArray *mexArray)
   /**********************************************************************************************/
   /* Fill in information gathered from the mxArray. For more information, see                   */
   /*   C/C++ and Fortran API Reference for Matlab:                                              */
   /*     http://www.mathworks.se/help/techdoc/apiref/bqoqnz0.html                               */
   /**********************************************************************************************/
   {

      dims     = mxGetNumberOfDimensions(mexArray);
      size     = mxGetNumberOfElements(mexArray);
      complex  = mxIsComplex(mexArray);
      mx_class = mxGetClassID(mexArray);

      shape    = new size_t[dims];
      const mwSize* mw_shape = mxGetDimensions(mexArray);
      for( size_t i=0; i<dims; ++i)
         shape[i] = mw_shape[i];
      
#ifdef VERBOSE
      printf("Dims = %d ( ", dims);
      for( size_t i=0; i<dims; ++i)
         printf("%d ", shape[i]);

      printf(")\n");
      printf("Size: %d\n",size);
#endif

  }

   size_t mmapMatlabToC(size_t i, const size_t *mshape=NULL)
   /**********************************************************************************************/
   /* This is a general method for mapping an array in linear Matlab/Fortran memory space to     */
   /* to linear C/Python space. It was a bitch to figure this one out, so treat it with respect! */
   /* It handles arrays of any number of dimensions and shape.                                   */
   /**********************************************************************************************/
   {
      size_t rate, I, stride;

      if( mshape == NULL )
        mshape = shape;

      rate   = 1;
      for( size_t d=1; d<dims; ++d )
         rate *= mshape[d];
      I      = 0;
      stride = 1;
      for( size_t d=0; d<dims-1; ++d ) {
         I      += stride*((i/rate)%mshape[d]);
         rate   /= mshape[d+1];
         stride *= mshape[d];
      }
      I += stride*((i/rate)%mshape[dims-1]);

#ifdef VERBOSE
      // Hardcoded formulas for memory mapping. Used to check the general method above.
      size_t Iref = 0;
      if(dims==1)
         Iref = i;
      else if(dims==2)
         Iref = (i%shape[1])*shape[0]+i/shape[1];
      else if(dims==3)
         Iref = ((i/1)%shape[2])*shape[1]*shape[0]+((i/shape[2])%shape[1])*shape[0]+i/(shape[2]*shape[1]);

      printf("Dims: %d   Computed index (reference): %d (%d)\n", dims, I, Iref );
#endif

      return I;
  }

  size_t mmapCToMatlab(size_t i)
  /**********************************************************************************************/
  /* Given the function above, how can we go the other way around, i.e., from C/Python memory   */
  /* memory space to Matlab/Fortran memory space by simply invoking the method above with the  */
  /* 'shape' variable inverted.                                                                 */
  /**********************************************************************************************/
  {
     size_t *inv_shape = new size_t[dims];
     for(size_t d=0; d<dims; ++d)
        inv_shape[d] = shape[dims-1-d];
     return mmapMatlabToC(i, inv_shape);
  }

  virtual ~MexArray()
  { delete[] shape; }

};

                                                /*************************************************/
///////////////////////////                     /* A class intended to handle the hassle of      */
// CLASS RealMexArray //                        /* handling real Matlab arrays of arbitrary      */
///////////////////////////                     /* shapes and types in a robust way.             */
                                                /*************************************************/

template <class T>
class RealMexArray: public MexArray
{
public:
  T *data;

   RealMexArray(size_t dims_in, const size_t *shape_in, mxClassID class_in): MexArray(dims_in, shape_in, class_in, false)
  /**********************************************************************************************/
  /* Construct RealMexArray based on user supplied parameters.                                  */
  /**********************************************************************************************/
   {
      data = new T[size];
      for( size_t i=0; i<size; ++i) {
         data[i] = 0;
      }
   }


   template <class V>
   void importData(V *mxRePtr)
   /**********************************************************************************************/
   /* Takes a copy of the data and realigns it from being Matlab-continous to be C-continuous    */
   /**********************************************************************************************/
   {
      size_t I;
      for( size_t i=0; i<size; ++i ) {
         I = mmapMatlabToC(i);

         data[i] = (T)mxRePtr[I];
      }
   }

   RealMexArray(const mxArray *mexArray): MexArray(mexArray)
   /**********************************************************************************************/
   /* Constructs the RealMexArray based on supplied mxArray                                      */
   /**********************************************************************************************/
   {
      if( complex )
         printf("RealMexArray(const mxArray *mexArray) - mexArray is complex!");
         //mexErrMsgTxt("RealMexArray(const mxArray *mexArray) - mexArray is complex!");

      data = new T[size];

      // Call up import function with a type that match the mxArrays classID
      switch(mxGetClassID(mexArray)) {
      case mxDOUBLE_CLASS:
         importData((double*)mxGetPr( mexArray )); break;
      case mxSINGLE_CLASS:
         importData((float*)mxGetPr( mexArray )); break;
      case mxUINT32_CLASS:
         importData((unsigned int*)mxGetPr( mexArray )); break;
      case mxINT32_CLASS:
         importData((int*)mxGetPr( mexArray )); break;
      case mxLOGICAL_CLASS:
         importData((bool*)mxGetPr( mexArray )); break;
      default:
         printf("ComplexMexArray(const mxArray *mexArray) - Unknown data type!!!");
         //mexErrMsgTxt("ComplexMexArray(const mxArray *mexArray) - Unknown data type!!!");
      }
   }


   template <class V>
   void exportData(mxArray* mexArray)
   /**********************************************************************************************/
   /* Takes a copy of the data and realigns it from being C-continous to be Matlab-continuous    */
   /**********************************************************************************************/
   {
      V *mxRePtr = (V*)mxGetPr(mexArray);

      size_t I;
      for( size_t i=0; i<size; ++i ) {
         I = mmapCToMatlab(i);
         mxRePtr[i] = (V)data[I];
      }
   }

   mxArray* copy() {
   /**********************************************************************************************/
   /* This function creates a copy of the data that may be assigned to a Matlab output mxArray.  */
   /**********************************************************************************************/

      // To support both 32 and 64 bit systems: Copy 'shape' to a new array of the same type as mwSize
      mwSize *mwshape = new mwSize[dims];
      for( int i=0; i<size; ++i ) {
         mwshape[i] = (mwSize)shape[i];
      }
      mxArray *mexArray = mxCreateNumericArray( (mwSize)dims, mwshape, mx_class, mxREAL );

      // Call up export function with a type that match the mxArrays classID
      switch(mx_class) {
      case mxDOUBLE_CLASS:
         exportData<double>(mexArray); break;
      case mxSINGLE_CLASS:
         exportData<float>(mexArray); break;
      case mxUINT32_CLASS:
         exportData<unsigned int>(mexArray); break;
      case mxINT32_CLASS:
         exportData<int>(mexArray); break;
      case mxLOGICAL_CLASS:
         exportData<bool>(mexArray); break;
      default:
         printf("ComplexMexArray(const mxArray *mexArray) - Unknown data type!!!");
         //mexErrMsgTxt("ComplexMexArray(const mxArray *mexArray) - Unknown data type!!!");
      }
      return mexArray;
   }

  virtual ~RealMexArray()
  { delete[] data; }

};
                                                /*************************************************/
  ///////////////////////////                   /* A class intented to handle the hassle of      */
 // CLASS ComplexMexArray //                    /* handling complex Matlab arrays of arbitrary   */
///////////////////////////                     /* shapes and types in a robust way.             */
                                                /*************************************************/

template <class T>
class ComplexMexArray: public MexArray
{
public:
   Complex<T> *data;

   ComplexMexArray(size_t dims_in, const size_t *shape_in, mxClassID class_in): MexArray(dims_in, shape_in, class_in, true)
   /**********************************************************************************************/
   /* Construct ComplexMexArray based on user supplied parameters.                               */
   /**********************************************************************************************/
   {
      data = new Complex<T>[size];

      for( size_t i=0; i<size; ++i) {
         data[i].re = 0;
         data[i].im = 0;
      }
   }

   template <class V>
   void importData(V *mxRePtr, V *mxImPtr)
   /**********************************************************************************************/
   /* Takes a copy of the data and realigns it from being Matlab-continous to be C-continuous    */
   /**********************************************************************************************/
   {
      size_t I;
      for( size_t i=0; i<size; ++i ) {
         I = mmapMatlabToC(i);

         data[i].re = (T)mxRePtr[I];
         data[i].im = (T)mxImPtr[I];
      }
   }

   ComplexMexArray(const mxArray *mexArray): MexArray(mexArray)
   /**********************************************************************************************/
   /* Constructs the ComplexMexArray based on supplied mxArray                                   */
   /**********************************************************************************************/
   {
      if( !complex )
         printf("ComplexMexArray(const mxArray *mexArray) - mexArray is real!");
         //mexErrMsgTxt("ComplexMexArray(const mxArray *mexArray) - mexArray is real!");

      data = new Complex<T>[size];

      // Call up import function with a type that match the mxArrays classID
      switch(mxGetClassID(mexArray)) {
      case mxDOUBLE_CLASS:
         importData((double*)mxGetPr( mexArray ), (double*)mxGetPi( mexArray )); break;
      case mxSINGLE_CLASS:
         importData((float*)mxGetPr( mexArray ), (float*)mxGetPi( mexArray )); break;
      case mxUINT32_CLASS:
         importData((unsigned int*)mxGetPr( mexArray ), (unsigned int*)mxGetPi( mexArray )); break;
      case mxINT32_CLASS:
         importData((int*)mxGetPr( mexArray ), (int*)mxGetPi( mexArray )); break;
      default:
         printf("ComplexMexArray(const mxArray *mexArray) - Unknown data type!!!");
         //mexErrMsgTxt("ComplexMexArray(const mxArray *mexArray) - Unknown data type!!!");
      }
   }

   template <class V>
   void exportData(mxArray* mexArray)
   /**********************************************************************************************/
   /* Takes a copy of the data and realigns it from being C-continous to be Matlab-continuous    */
   /**********************************************************************************************/
   {
      V *mxRePtr = (V*)mxGetPr(mexArray);
      V *mxImPtr = (V*)mxGetPi(mexArray);

      size_t I;
      for( size_t i=0; i<size; ++i ) {
         I = mmapCToMatlab(i);
         mxRePtr[i] = (V)data[I].re;
         mxImPtr[i] = (V)data[I].im;
      }
   }

   mxArray* copy()
   /**********************************************************************************************/
   /* This function creates a copy of the data that may be assigned to a Matlab output mxArray.  */
   /**********************************************************************************************/
   {
      // To support both 32 and 64 bit systems: Copy 'shape' to a new array of the same type as mwSize
      mwSize *mwshape = new mwSize[dims];
      for( size_t i=0; i<dims; ++i ) {
         mwshape[i] = (mwSize)shape[i];
      }
      mxArray *mexArray = mxCreateNumericArray( (mwSize)dims, mwshape, mx_class, mxCOMPLEX );

      // Call up export function with a type that match the mxArrays classID
      switch(mx_class) {
      case mxDOUBLE_CLASS:
         exportData<double>(mexArray); break;
      case mxSINGLE_CLASS:
         exportData<float>(mexArray); break;
      case mxUINT32_CLASS:
         exportData<unsigned int>(mexArray); break;
      case mxINT32_CLASS:
         exportData<int>(mexArray); break;
      default:
         mexErrMsgTxt("ComplexMexArray(const mxArray *mexArray) - Unknown data type!!!");
         //mexErrMsgTxt("ComplexMexArray(const mxArray *mexArray) - Unknown data type!!!");
      }
      return mexArray;
   }

  virtual ~ComplexMexArray()
  { delete[] data; }

};

#endif
