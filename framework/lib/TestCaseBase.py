'''
Created on 18. aug. 2011

@author: jpaasen
'''
import unittest
import numpy as np
from ..mypylab import *
from ..gfuncs import is_np_array_type

np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=300)
np.set_printoptions(threshold=10000)

#PLOT_DIFF = False
FLOATING_PRECISION = 32 #64 for double

class TestCaseBase(unittest.TestCase):
   
   def setUp(self):
      pass
   
   def debug(self, x_py, R_ans, R_py):
      print "hello"
      
      print "hello2 "
      
      
   def almostEqual(self, A, B, precision=None, plot_diff=False, just_report=False, use_numpy=True):
      if precision == None:
         if FLOATING_PRECISION == 32:
            precision = 7
         elif FLOATING_PRECISION == 64:
            precision = 16
         else:
            print "FLOATING_PRECISION has invalid value (neither 32 nor 64)."
            exit()

      if plot_diff and is_np_array_type(A.__class__) and is_np_array_type(B.__class__):
         fig = figure()
         ax = fig.add_subplot(1,1,1)
         cax = ax.imshow(A.squeeze().real-B.squeeze().real)
         colorbar(cax)
         savefig()
         fig = figure()
         ax = fig.add_subplot(1,1,1)
         cax = ax.imshow(A.squeeze().imag-B.squeeze().imag)
         colorbar(cax)
         savefig()
                
         

      # Unwrap the array object one dimension at a time:
      def dissembleAndTest(A,B,dim,loc,shape):
         for i in range(A.shape[0]):
            loc[dim] = i
            if A.shape.__len__() > 1:
               dissembleAndTest(A[i],B[i],dim+1,loc,shape)
            else:
               loc_str = ' '
               shape_str = ' '
               for j in range(dim+1):
                  loc_str   = loc_str   + "%d "%loc[j]
                  shape_str = shape_str + "%d "%shape[j]               
               
               if just_report:
                  if( (A[i].real-B[i].real > 10**(-precision)) or (A[i].imag-B[i].imag > 10**(-precision)) ):
                     print  "expected %f+%fj, was %f+%fj (diff %e+%ej). Index (%s) Shape (%s)"\
                            %(A[i].real,A[i].imag,B[i].real,B[i].imag,A[i].real-B[i].real, A[i].imag-B[i].imag,loc_str,shape_str)
                  
               else:
                  self.assertAlmostEqual(A[i], B[i], precision,
                     "expected %f+%fj, was %f+%fj (diff %e+%ej). Index (%s) Shape (%s)"\
                     %(A[i].real,A[i].imag,B[i].real,B[i].imag,A[i].real-B[i].real, A[i].imag-B[i].imag,loc_str,shape_str))

               
      if is_np_array_type(A.__class__) or is_np_array_type(B.__class__):
         if is_np_array_type(A.__class__) and is_np_array_type(B.__class__):
            if A.shape == B.shape:
               
               if use_numpy:
                  T = A-B
                  max_diff = np.max(T)
                  if max_diff > precision:
                     self.fail("Numpy arrays are not equal!!!")
                     
                  else:
                     dissembleAndTest(A,B,0,np.zeros(A.shape.__len__()),A.shape)
            else:
               self.fail("variables are not of the same type and/or shape")
         else:
            self.fail("variables are not of the same type and/or shape")
         
      else:
         self.assertEqual(A, B,
                          "expected %f+%fj, was %f+%fj (diff %e+%ej)"\
                           %(A.real,A.imag,B.real,B.imag,A.real-B.real, A.imag-B.imag))

   def equal(self, A, B):
      
      # Unwrap the array object one dimension at a time:
      def dissembleAndTest(A,B):
         for i in range(A.shape[0]):
            if A.shape.__len__() > 1:
               dissembleAndTest(A[i],B[i])
            else:
               self.assertEqual(A[i], B[i],
                       "expected %f+%fj, was %f+%fj (diff %e+%ej)"\
                       %(A[i].real,A[i].imag,B[i].real,B[i].imag,A[i].real-B[i].real, A[i].imag-B[i].imag))
               
      if is_np_array_type(A.__class__) or is_np_array_type(B.__class__):
         if is_np_array_type(A.__class__) and is_np_array_type(B.__class__):
            if A.shape == B.shape:
               dissembleAndTest(A,B)
            else:
               self.fail("variables are not of the same type and/or shape")
         else:
            self.fail("variables are not of the same type and/or shape")
         
      else:
         self.assertEqual(A, B,
                          "expected %f+%fj, was %f+%fj (diff %e+%ej)"\
                           %(A.real,A.imag,B.real,B.imag,A.real-B.real, A.imag-B.imag))
         
