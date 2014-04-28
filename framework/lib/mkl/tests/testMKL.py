'''
Created on 18. aug. 2011

@author: jpaasen
'''
import unittest, sys, os
#import testComplexH as TC
import framework.mynumpy as np

from framework.lib.TestCaseBasec import TestCaseBase
from framework.lib.mkl.mklUnivariateSplineC import mklcUnivariateSplineC
from framework.lib.mkl.mklcSolveCholeskyC import mklcSolveCholeskyC

np.set_printoptions(precision=8)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=300)
np.set_printoptions(threshold=10000)

import time


FLOATING_PRECISION = 32 #64 for double
VERBOSE = False
DEBUG   = False

class TestMKL(TestCaseBase):
 
   def setUp(self):
      pass
                
   def testComplexUniformUnivariateSpline(self):
      
      x = np.arange(10)
      y = x**2 + 1j*(x**2+1)
      xi = np.linspace(0,9,91)
      #
      
      yi = mklcUnivariateSplineC( x, y, xi, 0, True )
      
      print yi
      
   def testComplexNonUniformUnivariateSpline(self):
      
      x = np.arange(10)
      y = x**2 + 1j*(x**2+1)
      xi = np.linspace(0,9,91)
      #
      
      yi = mklcUnivariateSplineC( x, y, xi, 0, False )
      
      print yi


   def testComplexCholeskySolver(self):
      
      N = 32
      x = np.random.normal(loc=0.0, scale=1.0, size=(N,)) + 1j*np.random.normal(loc=0.0, scale=1.0, size=(N,))
      A = np.outer(x,x.conj()).astype('complex128')
      A = A + 0.5*np.eye(N)
      b = np.ones(N,dtype=np.complex128)
   
#      A = np.array([[3+0j,5+1j],[5-1j,14+0j]])
#      b = np.array([1+0j,1+0j])
   
      yi = mklcSolveCholeskyC( A, b )
      
      
      print yi
      
      print 'Numpy reference:'
      
      C = np.linalg.solve( A, b )
      print C
      
      print 'hello'


def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestMKL)
