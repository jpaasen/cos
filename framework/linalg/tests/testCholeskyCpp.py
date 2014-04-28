'''
Created on 18. aug. 2011

@author: jpaasen
'''
import unittest

import scipy as sp
#from scipy.linalg import cho_solve, cho_factor

#import numpy as np
import framework.mynumpy as np
#import testMatrixOperations as tmo
import framework.linalg.matrixDecomp as md
#import framework.mynumpy as mynp
import framework.linalg.linearSolve as ls 
from framework.gfuncs import is_np_type

FLOATING_PRECISION = 32 #64 for double

class TestCholeskyCpp(unittest.TestCase):

   def setUp(self):
      self.n = 5
      self.R = np.zeros((self.n,self.n),dtype=complex)
      self.a = np.zeros((self.n,),dtype=complex)
      mu = 2
      std = 3
      for i in range(self.R.shape[1]):
         self.a[i] = complex(np.random.normal(mu,std),np.random.normal(mu,std))
         for j in range(self.R.shape[0]):
            self.R[j,i] = complex(np.random.normal(mu,std),np.random.normal(mu,std))


   def almostEqual(self, A, B, precision=15):
      if FLOATING_PRECISION == 32:
         precision = 7
      elif FLOATING_PRECISION == 64:
         precision = 16
      else:
         print "FLOATING_PRECISION has invalid value (neither 32 nor 64)."
         exit()
         

      # Unwrap the array object one dimension at a time:
      def dissembleAndTest(A,B):
         for i in range(A.shape[0]):
            if A.shape.__len__() > 1:
               dissembleAndTest(A[i],B[i])
            else:
               self.assertAlmostEqual(A[i], B[i], precision,
                       "expected %f+%fj, was %f+%fj (diff %e+%ej)"\
                       %(A[i].real,A[i].imag,B[i].real,B[i].imag,A[i].real-B[i].real, A[i].imag-B[i].imag))
               
      if is_np_type(A.__class__):
         if is_np_type(B.__class__) and A.shape == B.shape:
            dissembleAndTest(A,B)
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
               
      if is_np_type(A.__class__):
         if is_np_type(B.__class__) and A.shape == B.shape:
            dissembleAndTest(A,B)
         else:
            self.fail("variables are not of the same type and/or shape")
         
      else:
         self.assertEqual(A, B,
                          "expected %f+%fj, was %f+%fj (diff %e+%ej)"\
                           %(A.real,A.imag,B.real,B.imag,A.real-B.real, A.imag-B.imag))
                       
   def testCholesky(self):
      
#      cho_solve(cho_factor(R), a)
      self.R = np.array([[3,2],[2,3]],dtype=complex)
      self.a = np.array([7,7],dtype=complex)
      self.n = 2
      
      Ria = np.linalg.solve(self.R, self.a)
      
      
      U = md.cholesky(self.R, self.n)
      UT = U.conjugate().transpose()
      
      y = ls.forwardSolve(UT, self.a, self.n)
      x = ls.backtrackSolve(U, y, self.n)
   
      
      self.almostEqual(Ria, x)
      
#   U = md.cholesky(A, n)
#
#   UT = np.conjugatetranspose(U)
#   
#   y = forwardSolve(UT, b, n)
#   x = backtrackSolve(U, y, n)
#   
#   return x


               
   
def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestCholeskyCpp)
