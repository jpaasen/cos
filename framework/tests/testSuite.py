
import sys, os
#sys.path.append("%s/python"%os.environ['PHDCODE_ROOT'])
#sys.path.append("%s/python/framework/linalg"%os.environ['PHDCODE_ROOT'])
#sys.path.append("%s/python/framework/beamformer/capon"%os.environ['PHDCODE_ROOT'])
#sys.path.append("%s/../"%os.getcwd())
#print "%s/../"%os.getcwd()

import unittest

import framework.beamformer.capon.tests.testRTC as testRTC


def suite():
   
#   raw_input()
   
   suites = []
    
   suites.append(testRTC.suite())
   
   return unittest.TestSuite(suites)


def load_tests(loader, tests, pattern): 
   return suite()


if __name__ == "__main__":
   #import sys;sys.argv = ['', 'Test.testName']
   suites = suite()
   unittest.TextTestRunner(verbosity=2).run(suites)
