'''
Created on 30. aug. 2011

@author: jpaasen
'''
#import os
#execfile('%s/python/setup_environment.py'%os.environ['PHDCODE_ROOT'])

import unittest

#from framework.tests.testlinalg import *

import testCholesky as tChol
import testMatrixUtils as tMatUtil
import testLinearSolve as tLinSolve
import testBeamspace as tBs
import testWoodbury as tw
import testUHDU as tuhdu

# testing cython module
import testLinearSolvec as tLinSolvec

# testing c++ module
import testCholeskyCpp as tCholCpp

def suite():
   
   suites = []
   
   suites.append(tMatUtil.suite())
   suites.append(tChol.suite())
   suites.append(tLinSolve.suite())
   suites.append(tBs.suite())
   suites.append(tw.suite())
   suites.append(tuhdu.suite())
   
   suites.append(tLinSolvec.suite())
   
   suites.append(tCholCpp.suite())
   
   return unittest.TestSuite(suites)


def load_tests(loader, tests, pattern): 
   return suite()


if __name__ == "__main__":
   #import sys;sys.argv = ['', 'Test.testName']
   suites = suite()
   unittest.TextTestRunner(verbosity=2).run(suites)
