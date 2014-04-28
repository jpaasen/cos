'''
Created on 18. aug. 2011

@author: jpaasen
'''
import unittest

import testMatrixOperations as tmo
import framework.linalg.matrixDecomp as md
import framework.mynumpy as mynp
import framework.linalg.linearSolve as ls 

class TestCholesky(tmo.TestMatrixOperations):

              
   def testIfCholeskyRIsAsExpected(self):
      
      R = md.cholesky(self.A, self.n)
      
      for i in range(self.n):
         for j in range(self.n):   
            self.assertAlmostEqual(self.R[i][j].tolist(), R[i][j].tolist(), 14, 
                              "expected " + str(self.R[i][j]) + " was " + str(R[i][j].real))
               
            
   def testRTransposedTimesRIsEqualA(self):
      
      R = md.cholesky(self.A, self.n)
      
      RT = mynp.conjugatetranspose(R)
      
      A = mynp.dot(RT, R)
      
      for i in range(self.n):
         for j in range(self.n):   
            self.assertAlmostEqual(self.A[i][j].tolist(), A[i][j].tolist(), 15, 
                              "expected " + str(self.A[i][j]) + " was " + str(A[i][j]))
         
   
   def testSolvingAxbWithCholeskyAndForwardBackwardSolve(self):
      
      R = md.cholesky(self.A, self.n) 
      
      RT = mynp.conjugatetranspose(R)
      
      y = ls.forwardSolve(RT, self.b1, self.n)
      x = ls.backtrackSolve(R, y, self.n)
      
      for i in range(self.n):
         self.assertAlmostEqual(self.x1[i].tolist(), x[i].tolist(), 14, 
                           "expected " + str(self.x1[i]) + " was " + str(x[i]))
         
      y = ls.forwardSolve(RT, self.b2, self.n)
      x = ls.backtrackSolve(R, y, self.n)
      
      for i in range(self.n):
         self.assertAlmostEqual(self.x2[i].tolist(), x[i].tolist(), 14, 
                           "expected " + str(self.x2[i]) + " was " + str(x[i]))
       

   def testSolvingComplexAxbWithCholeskyAndForwardBackwardSolve(self):
      
      R = md.cholesky(self.complexA, self.n)
      
      for i in range(self.n):
         for j in range(self.n):   
            self.assertAlmostEqual(self.complexR[i][j].tolist(), R[i][j].tolist(), 14, 
                              "expected " + str(self.complexR[i][j]) + " was " + str(R[i][j]))

      RT = mynp.conjugatetranspose(R)
      
      y = ls.forwardSolve(RT, self.complexb, self.n)
      
      for i in range(self.n):
         self.assertAlmostEqual(self.complexy[i].tolist(), y[i].tolist(), 14, 
                           "expected " + str(self.complexy[i]) + " was " + str(y[i]))
      
      x = ls.backtrackSolve(R, y, self.n)
      
      for i in range(self.n):
         self.assertAlmostEqual(self.complexx[i].tolist(), x[i].tolist(), 14, 
                           "expected " + str(self.complexx[i]) + " was " + str(x[i]))
         
   def testCholeskyInPlace(self):
      
      L = md.choleskyInPlace(self.complexA, self.n)
      R = mynp.conjugatetranspose(L)
      
      self.assertMatrixAlmosteEqual(self.complexR.tolist(), R.tolist(), 14)      

def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestCholesky)
