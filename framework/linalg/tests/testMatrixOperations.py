'''
Created on 29. aug. 2011

@author: jpaasen
'''
import unittest

import framework.mynumpy as np

class TestMatrixOperations(unittest.TestCase):

   def setUp(self):
      
      self.n = 3
      
      self.A = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]],dtype=complex)
      self.A2 = np.array([[4.0, -2.0, 0.0], [-2.0, 4.0, -2.0], [0.0, -2.0, 4.0]],dtype=complex)
      self.R = np.array([[1.414213562373095, -0.707106781186547, 0.0],
              [0.0, 1.224744871391589, -0.816496580927726],
              [0.0, 0.0, 1.154700538379252]],dtype=complex)
      
      self.AA = np.array([[5, -4, 1], [-4, 6, -4], [1, -4, 5]],dtype=complex)
      self.B = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]],dtype=complex)
      self.BT = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]],dtype=complex)
      
      self.b1 = np.array([1.0, 1.0, 1.0],dtype=complex)
      self.x1 = np.array([3.0/2, 2.0, 3.0/2],dtype=complex)

      self.b2 = np.array([1, 2, 3],dtype=complex)
      self.x2 = np.array([5.0/2, 4, 7.0/2],dtype=complex)
      
      self.C = np.array([[1, 2, 3], [0, 1, 1], [0, 0, 1]],dtype=complex)
      self.b1c = np.array([1, 1, 1],dtype=complex)
      self.x1c = np.array([-2, 0, 1],dtype=complex)
      self.x1cT = np.array([1, -1, -1],dtype=complex)
      
      self.x0_zero = np.zeros((3,), dtype=complex)
      
      self.Ab2 = np.array([0, 0, 4],dtype=complex)
      self.Ab1 = np.array([1, 0, 1],dtype=complex)
      
      self.Ab1b1T = np.array([[3, 1, 3], [1, 6, 5], [3, 5, 11]],dtype=complex)
      self.invA = np.array([[0.750, 0.50, 0.250], [0.50, 1.0, 0.50], [0.250, 0.50, 0.750]],dtype=complex)
      self.invAb1b1T = np.array([[0.465909090909091, 0.045454545454545, -0.147727272727273],
                        [0.045454545454545, 0.272727272727273, -0.136363636363636],
                        [-0.147727272727273, -0.136363636363636, 0.193181818181818]],dtype=complex)
      
      self.complexA = np.array([[2.0, 3.0 + 1.0j, 2.0 - 2.0j], 
                   [3.0 - 1.0j, 9.0, -2.0j], 
                   [2.0 + 2.0j, 2.0j, 14.0]], dtype=complex)
      self.complexR = np.array([[1.414213562373095, 2.121320343559642 + 0.707106781186547j, 1.414213562373095 - 1.414213562373095j], 
                   [0.0, 2.0, -1.0 + 1.0j], 
                   [0.0, 0.0, 2.828427124746190]], dtype=complex)
      self.complexb = np.array([1.0, 1.0 + 1.0j, 1.0 - 2.0j], dtype=complex)
      self.complexy = np.array([0.707106781186547 - 0.0j,
                   -0.250 + 0.750j,
                   -0.353553390593273 - 0.883883476483184j], dtype=complex)
      self.complexx = np.array([1.593749999999999 - 0.06250j,
                   -0.343750 + 0.281250j,
                   -0.1250 - 0.31250j], dtype=complex)
      self.complexAb = np.array([2.0 - 2.0j, 8.0 + 6.0j, 14.0 - 24.0j], dtype=complex)
      
      self.but4 = np.array([[0.5]*4, [0.5, -0.5j, -0.5, 0.5j], [0.5, -0.5]*2, [0.5, 0.5j, -0.5, -0.5j]], dtype=complex)
      self.but3 = np.array([[0.577350269189626, 0.577350269189626, 0.577350269189626],                    
                   [0.577350269189626, -0.288675134594813 - 0.50j, -0.288675134594813 + 0.50j],               
                   [0.577350269189626, -0.288675134594813 + 0.50j, -0.288675134594813 - 0.50j]], dtype=complex)
      self.bsComplexb = np.array([1.732050807568878 - 0.577350269189626j, 1.50 + 0.288675134594813j], dtype=complex)
      
      self.diag = 0.2
      self.x = np.array([1.0, 1.0 + 1.0j, 1.0 - 2.0j, 2.0 + 1.0j], dtype=complex)
      
      self.complexAbbH = np.array([[10.0, 11.0 + 1.0j, 10.0 - 2.0j],
                          [11.0 - 1.0j, 17.0, 8.0 - 2.0j],
                          [10.0 + 2.0j, 8.0 + 2.0j, 22.0]], dtype=complex)
      self.complexInvAbbH = np.array([[1.067010309278351, -0.407216494845361 - 0.015463917525773j, -0.190721649484536 + 0.020618556701031j],
                             [-0.407216494845361 + 0.015463917525773j, 0.247422680412371, 0.077319587628866 - 0.015463917525773j],
                             [-0.190721649484536 - 0.020618556701031j,  0.077319587628866 + 0.015463917525773j,  0.087628865979381]], dtype=complex)  
      self.complexInvA = np.array([[1.906250, -0.593750 - 0.156250j,  -0.250 + 0.18750j],
                          [-0.593750 + 0.156250j,  0.31250, 0.06250 - 0.06250j],                    
                          [-0.250 - 0.18750j,  0.06250 + 0.06250j, 0.1250]], dtype=complex)
      
      self.complexA4x4 = np.array([[22.0, 8.0, 11.0 - 11.0j, 22.0 - 7.0j],
                          [8.0, 22.0, 17.0 - 2.0j, 11.0 - 7.0j],
                          [11.0 + 11.0j, 17.0 + 2.0j, 45.0, 23.0 - 5.0j],
                          [22.0 + 7.0j, 11.0 + 7.0j, 23.0 + 5.0j, 37.0]], dtype=complex)
      self.U4x4 = np.array([[1.0000, 0.3636, 0.50 - 0.50j, 1.0 - 0.3182j],
                   [0.0, 1.0, 0.6810 + 0.1048j, 0.1571 - 0.2333j],
                   [0.0, 0.0, 1.0, 0.2776 - 0.3670j],
                   [0.0, 0.0, 0.0, 1.0]], dtype=complex)
      self.D4x4 = np.array([22.0, 19.0909, 24.9381, 5.9806])
      
      self.sonardata_R = np.array(np.load('./data/data_R.npy')) # created without diagonal loading
      self.sonardata_a = np.array(np.load('./data/data_a.npy'))
      self.sonardata_Ria = np.array(np.load('./data/data_Ria.npy'))
      self.sonardata_ar = np.array(np.load('./data/data_ar.npy'))
      self.sonardata_n = 32
      
      # random data for testing
      self.L = L = 24
      self.d = d = 100
      U = np.triu(np.random.randn(L,L) + np.random.randn(L,L)*1j) + np.eye(L)*d
      self.randA = np.dot(U.conjugate().T, U) 
      self.randb = np.random.randn(L) + np.random.randn(L)*1j 


   def tearDown(self):
      pass
      
   def assertMatrixAlmosteEqual(self, first, second, places):
         
      lenFirst = len(first)
      lenSecond = len(second)
         
      if lenFirst != lenSecond:
         self.fail('First and second list have different number of row elements')
         
      i,j = 0,0

      for row in first:   
         
         secRow = second[i]
         
         try:
            lenFirst = len(row)
         except:
            lenFirst = 1
            row = [row]
            
         try:   
            lenSecond = len(secRow)
         except:
            lenSecond = 1
            secRow = [secRow]
         
         if lenFirst != lenSecond:
            self.fail('First and second list have different number of column elements')
           
         j = 0
         for rowElem in row:
            
            secElem = secRow[j]
            
            self.assertAlmostEqual(rowElem.real, secElem.real, places, 
                                'Expected %.16f was %.16f (diff %e)' %(rowElem.real,secElem.real,rowElem.real-secElem.real))
            self.assertAlmostEqual(rowElem.imag, secElem.imag, places, 
                                'Expected ' + str(rowElem.imag) + ' was ' + str(secElem.imag))
            
            j = j + 1
         
         i = i + 1
         
