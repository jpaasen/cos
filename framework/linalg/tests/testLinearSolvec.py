'''
Created on 29. aug. 2011

@author: jpaasen
'''

import unittest

import testMatrixOperations as tmo
import framework.linalg.linearSolvec as ls
import framework.mynumpy as np

class TestLinearSolve(tmo.TestMatrixOperations):
   
   def testSolveBiCG(self):
      
      x = ls.solveBiCG(self.A, self.b1, self.x0_zero, 0, 0)
      self.assertMatrixAlmosteEqual(self.x1, x, 14)
      
      x = ls.solveBiCG(self.complexA, self.complexb, self.x0_zero, 0, 0)    
      self.assertMatrixAlmosteEqual(self.complexx, x, 13)
      
      x_ref = np.linalg.solve(self.randA, self.randb)
      x = ls.solveBiCG(self.randA, self.randb, np.zeros(self.L, dtype=complex), 0, 0)                   
      self.assertMatrixAlmosteEqual(x_ref, x, 15)
   
   def testSolveCholesky(self):
      
      x = ls.solveCholesky(self.A, self.b1, self.n) 
      self.assertMatrixAlmosteEqual(self.x1, x, 14)
        
      x = ls.solveCholesky(self.complexA, self.complexb, self.n)    
      self.assertMatrixAlmosteEqual(self.complexx, x, 14)
      
      x_ref = np.linalg.solve(self.randA, self.randb)
      x = ls.solveCholesky(self.randA, self.randb, self.L)
      self.assertMatrixAlmosteEqual(x_ref, x, 15)
      
         
   def testSolveUHDU(self):
      
      x = ls.solveUHDU(self.A, self.b1, self.n)
      self.assertMatrixAlmosteEqual(self.x1, x, 14)
      
      x = ls.solveUHDU(self.complexA, self.complexb, self.n) 
      self.assertMatrixAlmosteEqual(self.complexx, x, 14)
      
      x_ref = np.linalg.solve(self.randA, self.randb)
      x = ls.solveUHDU(self.randA, self.randb, self.L)                   
      self.assertMatrixAlmosteEqual(x_ref, x, 15)
      
   
   def testBacktrackSolve(self):
      
      x = ls.backtrackSolve(self.C, self.b1c, self.n)
      self.assertListEqual(self.x1c.tolist(), x.tolist(), "Error in backtrack solve")
      
   
   def testForwardSolve(self):
      
      x = ls.forwardSolve(np.conjugatetranspose(self.C), self.b1c, self.n)
      self.assertListEqual(self.x1cT.tolist(), x.tolist(), "Error in forward solve")
      
   
def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestLinearSolve)
