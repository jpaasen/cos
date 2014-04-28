'''
Created on 29. aug. 2011

@author: jpaasen

Does not make much sense to test numpy, but before this was testing a custom matrix library.
However, there are some custom function in mynumpy that are tested here.
'''
import unittest

import testMatrixOperations as tmo
import framework.mynumpy as mynp

class TestMatrixUtils(tmo.TestMatrixOperations):
   

   def testMatrixMulSquareATimesA(self):
      
      C = mynp.dot(self.A, self.A)    
      self.assertEqual(self.AA.tolist(), C.tolist())
      
            
   def testMatrixVectorMul(self):
      
      errstr = 'Error in result returned by matrix vector multiplication'
      
      x = mynp.dot(self.A, self.b2)
      self.assertListEqual(x.tolist(), self.Ab2.tolist(), errstr + ' (Square matrix)')

      x = mynp.dot(self.A, self.b1)
      self.assertListEqual(x.tolist(), self.Ab1.tolist(), errstr + ' (Square matrix)')

      x = mynp.dot(self.A[0:2], self.b2)
      self.assertListEqual(x.tolist(), self.Ab2[0:2].tolist(), errstr + ' (non-square matrix)')


   def testMatrixMul(self):
      
      errstr = 'Error in result returned by matrix vector multiplication'
      
      C = mynp.dot(self.A, self.A)    
      self.assertListEqual(self.AA.tolist(), C.tolist())
      
      x = mynp.dot(self.A, self.b2)
      self.assertListEqual(x.tolist(), self.Ab2.tolist(), errstr + ' (Square matrix)')

      x = mynp.dot(self.A, self.b1)
      self.assertListEqual(x.tolist(), self.Ab1.tolist(), errstr + ' (Square matrix)')

      x = mynp.dot(self.A[0:2], self.b2)
      self.assertListEqual(x.tolist(), self.Ab2[0:2].tolist(), errstr + ' (non-square matrix)')
      
      
   def testMatrixAdd(self):
      
      C = self.A + self.A
      self.assertListEqual(self.A2.tolist(), C.tolist(), 'Error in matrix add')
      
   
   def testMatrixScalarMul(self):
      
      C = self.A * 2
      self.assertListEqual(self.A2.tolist(), C.tolist(), 'Error in matrix scalar mul')
   

   def testComplexMatrixVectorMul(self):
      
      errstr = 'Error in result returned by matrix vector multiplication'
      
      x = mynp.dot(self.complexA, self.complexb)
      self.assertListEqual(x.tolist(), self.complexAb.tolist(), errstr + ' (square complex matrix)')
      
      x = mynp.dot(self.complexA[0:2], self.complexb)
      self.assertListEqual(x[0:2].tolist(), self.complexAb[0:2].tolist(), errstr + ' (non-square complex matrix)')


   def testTransposed(self):
      
      BT = mynp.transpose(self.B)    
      self.assertListEqual(self.BT.tolist(), BT.tolist())
   
   
   def testComplexConjugateTransposed(self):
      
      AT = mynp.conjugatetranspose(self.complexA)
      self.assertListEqual(self.complexA.tolist(), AT.tolist())
         
            
def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestMatrixUtils)
