
cdef extern from "Complex.h":
   cdef cppclass Complex[T]:
      Complex()
      Complex(float, float)
      Complex[T] operator+(Complex[T]&)
      Complex[T] operator*(Complex[T]&)
      
