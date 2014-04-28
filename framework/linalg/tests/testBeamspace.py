'''
Created on 31. aug. 2011

@author: jpaasen
'''

import unittest
import testMatrixOperations as tmo
import framework.linalg.beamspace as bs

class TestBeamspace(tmo.TestMatrixOperations):
   
   
   def testButlerMatrixGeneration(self):
      
      B = bs.butlerMatrix(4, 4)
      self.assertMatrixAlmosteEqual(self.but4.tolist(), B.tolist(), 15)
            
      C = bs.butlerMatrix(3, 3)
      self.assertMatrixAlmosteEqual(self.but3.tolist(), C.tolist(), 15)
            
   
   def testBeamspaceTransform(self):
      
      m,n = 2,3
      B = bs.butlerMatrix(m, n)
      b = bs.beamspace(B, self.complexb)
      
      self.assertMatrixAlmosteEqual(self.bsComplexb.tolist(), b.tolist(), 14)
      

def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestBeamspace)
