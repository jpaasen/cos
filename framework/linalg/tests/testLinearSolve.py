'''
Created on 29. aug. 2011

@author: jpaasen
'''

import unittest

import testMatrixOperations as tmo
import framework.linalg.linearSolve as ls
import framework.mynumpy as mynp

class TestLinearSolve(tmo.TestMatrixOperations):
   
   def testSolveBiCG(self):
      
      n = len(self.b1)
      x0 = mynp.zeros(n, dtype=complex)
      
      x = ls.solveBiCG(self.A, self.b1, x0, 0, n)
      self.assertMatrixAlmosteEqual(self.x1, x, 14)
      
      
      n = len(self.complexb)
      
      x = ls.solveBiCG(self.complexA, self.complexb, x0, 0, n)    
      self.assertMatrixAlmosteEqual(self.complexx, x, 12)
      
   
   def testSolveCholesky(self):
      
      x = ls.solveCholesky(self.A, self.b1, self.n) 
      self.assertMatrixAlmosteEqual(self.x1, x, 14)
        
      x = ls.solveCholesky(self.complexA, self.complexb, self.n)    
      self.assertMatrixAlmosteEqual(self.complexx, x, 14)
      
      x_ref = mynp.linalg.solve(self.randA, self.randb)
      x = ls.solveCholesky(self.randA, self.randb, self.L)                     
      self.assertMatrixAlmosteEqual(x_ref, x, 15)
      
         
   def testSolveUHDU(self):
      
      x = ls.solveUHDU(self.A, self.b1, self.n)
      self.assertMatrixAlmosteEqual(self.x1, x, 14)
      
      x = ls.solveUHDU(self.complexA, self.complexb, self.n) 
      self.assertMatrixAlmosteEqual(self.complexx, x, 14)
      
   
   def testSolveUHDUvsCholesky(self):
      
      x_uhdu = ls.solveUHDU(self.randA, self.randb, self.L)
      x_chol = ls.solveCholesky(self.randA, self.randb, self.L)
                       
      self.assertMatrixAlmosteEqual(x_uhdu, x_chol, 15)
      
   
   def testBacktrackSolve(self):
      
      x = ls.backtrackSolve(self.C, self.b1c, self.n)
      self.assertListEqual(self.x1c.tolist(), x.tolist(), "Error in backtrack solve")
      
   
   def testForwardSolve(self):
      
      x = ls.forwardSolve(mynp.conjugatetranspose(self.C), self.b1c, self.n)
      self.assertListEqual(self.x1cT.tolist(), x.tolist(), "Error in forward solve")
      
   
def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestLinearSolve)
