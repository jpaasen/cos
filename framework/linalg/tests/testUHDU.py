'''
Created on 14. sep. 2011

@author: Hybel
'''
import unittest

import testMatrixOperations as tmo
import framework.linalg.matrixDecomp as md

class TestUHDU(tmo.TestMatrixOperations):
   
   def testUHDUDecomposition(self):
      
      UD = md.uhdu(self.complexA4x4, 4)
      
      self.assertMatrixAlmosteEqual(self.U4x4.tolist(), UD[0].tolist(), 4)
      self.assertMatrixAlmosteEqual(self.D4x4.tolist(), UD[1].tolist(), 4)
        
   
def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestUHDU)
