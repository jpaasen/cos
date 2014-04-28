#ifndef COMPLEX_H
#define COMPLEX_H

//#include "math.h"
#include <iostream>
                                                /*************************************************/
        //////////////////////                  /* Basic struct that defines the datatype that   */
       /// struct Complex ///                   /* is complex numbers. Also contain operator     */
      //////////////////////                    /* overloads.                                    */
                                                /*************************************************/
template <class T>
class Complex
{ public:
  T re, im;                              //Stores the information about a complex number
                                                //in rectangular form.
  Complex()
  { re = im = 0.0;                            //At creation we reset the number.
  }
  Complex(T r, T i)
  { re = r;
    im = i;                            //At creation we reset the number.
  }
  T abs()
  { return(sqrt(re * re + im * im));        //It is often practical knowing the absolute value.
  }
  T abs2()
  {  return(re * re + im * im);
  }
  Complex<T> conjugate()
  { return Complex<T>(re,-im);
  }
  Complex<T> conj()
  { return Complex<T>(re,-im);
  }
  T Re()
  { return re;                                 //We might also wish to know the real part
  }
  T Im()
  { return im;                                 //or the imaginary part.
  }
  
    //-------------------//
   // Storing operators //
  //-------------------//

//  inline Complex<T>& operator =  (Complex<T> &zIn)
//  { return zIn;
//  }
  inline Complex<T>* operator =  (Complex<T> *zIn)
  {  re = zIn->re;
     im = zIn->im;
     return zIn;
  }
  inline void operator =  (T dInitialValue)       //In case we recieve a double we assume it s a real
  { re = dInitialValue; im = 0.0;             //value and reset the imaginary part.
  }

    //----------------------//
   // Arithmetic operators //
  //----------------------//

  inline Complex<T> operator +  (Complex<T> &zIn)             //Returns the sum of two complex numbers.
  { Complex<T> zRes;
    zRes.re = re + zIn.re;
    zRes.im = im + zIn.im;
    return zRes;
  }
  inline void    operator += (Complex<T> &zIn)             //Adds the complex number on both sides of the
  { re += zIn.re;                             //operator together and stores the result in the
    im += zIn.im;                             //complex number on the left side.
  }
  inline Complex<T> operator -  (Complex<T> &zIn)             //Returns the difference between two complex
  { Complex<T> zRes;                               //numbers.
    zRes.re = re - zIn.re;
    zRes.im = im - zIn.im;
    return zRes;
  }
  inline void    operator -= (Complex<T> &zIn)             //Subtracts the right side complex number from the
  { re -= zIn.re;                             //left side complex number, and stores result in
    im -= zIn.im;                             //the left side complex number.
  }
  inline Complex<T> operator *  (Complex<T> &zIn)             //Returns the product of two complex numbers.
  { Complex<T> zRes;
    zRes.re = re*zIn.re - im*zIn.im;
    zRes.im = im*zIn.re + re*zIn.im;
    return zRes;
  }
  inline void    operator *= (Complex<T> &zIn)             //Multiplies both complex numbers together and
  { double fReCopy = re;
    re = re*zIn.re - im*zIn.im;            //stores product in the left side complex number.
    im = im*zIn.re + fReCopy*zIn.im;
  }
  inline Complex<T> operator *  (int &iScalar)             //Returns the scaled complex number.
  { Complex<T> zRes;
    zRes.re = re*iScalar;
    zRes.im = im*iScalar;
    return zRes;
  }
  inline Complex<T> operator *  (T &dScalar)          //Returns the scaled complex number.
  { Complex<T> zRes;
    zRes.re = re*dScalar;
    zRes.im = im*dScalar;
    return zRes;
  }
  inline void    operator *= (int &iScalar)             //Scales the left side complex number by the factor
  { re *= iScalar;                             //on the right side, and stores the product in the
    im *= iScalar;                             //left side complex number.
  }
  inline void    operator *= (T &dScalar)          //Scales the left side complex number by the factor
  { re *= dScalar;                             //on the right side, and stores the product in the
    im *= dScalar;                             //left side complex number.
  }
  inline Complex<T> operator /  (Complex<T> &zIn)             //Left side complex number is the dividend and right
  { Complex<T> zRes;                               //side divisor. Quotient is returned.
    double fTmp = zIn.re*zIn.re + zIn.im*zIn.im;
    zRes.re = (re*zIn.re + im*zIn.im)/fTmp;
    zRes.im = (im*zIn.re - re*zIn.im)/fTmp;
    return zRes;
  }
//  static inline void div(Complex<T> &A, Complex<T> &B, Complex<T> &R)
//  { //Complex zRes;                               //side divisor. Quotient is returned.
//    T fTmp = B.re*B.re + B.im*B.im;
//    R.re = (A.re*B.re + A.im*B.im)/fTmp;
//    R.im = (A.im*B.re - A.re*B.im)/fTmp;
//  //  return Complex<T>((re*zIn.re + im*zIn.im)/fTmp,(im*zIn.re - re*zIn.im)/fTmp);
//  }
  inline void    operator /= (Complex<T> &zIn)             //Left side complex number is the dividend and right
  { double fTmp = zIn.re*zIn.re + zIn.im*zIn.im; //side divisor. Quotient is stored in the right side
    double fReCopy = re;
    re = (re*zIn.re + im*zIn.im)/fTmp;     //complex number.
    im = (im*zIn.re - fReCopy*zIn.im)/fTmp;
  }
  inline Complex<T> operator /  (int &iScalar)             //Left side complex number is the dividend which
  { Complex<T> zRes;                               //is inversely scaled by the scalar divisor.
    zRes.re = re/iScalar;                     //Quotient is returned.
    zRes.im = im/iScalar;
    return zRes;
  }
  inline Complex<T> operator /  (T &dScalar)          //Left side complex number is the dividend which
  { Complex<T> zRes;                               //is inversely scaled by the scalar divisor.
    zRes.re = re/dScalar;                     //Quotient is returned.
    zRes.im = im/dScalar;
    return zRes;
  }
  inline void    operator /= (int &iScalar)             //Left side complex number is the dividend which
  { re /= iScalar;                             //is inversely scaled by the scalar divisor.
    im /= iScalar;                             //Quotient is stored in left side complex number.
  }
  inline void    operator /= (T &dScalar)          //Left side complex number is the dividend which
  { re /= dScalar;                             //is inversely scaled by the scalar divisor.
    im /= dScalar;                             //Quotient is stored in left side complex number.
  }
    
    //-------------------//
   // Logical operators //
  //-------------------//

  inline bool    operator == (Complex<T> &zIn)             //Compares two complex numbers, returns true if
  { return ( (re == zIn.re) && (im == zIn.im) );//they're alike.
  }
  inline bool    operator != (Complex<T> &zIn)             //Compares two complex numbers, returns false if
  { return ( (re != zIn.re) || (im != zIn.im) );//they're alike.
  }
};

#endif
