'''
Created on 2. sep. 2011

@author: jpaasen
'''
import unittest

import testMatrixOperations as tmo
import framework.linalg.woodbury as wb
import framework.mynumpy as mynp

class TestWoodbury(tmo.TestMatrixOperations):
   
   def testShermannMorrisonUp(self):

      invAxxH = wb.shermanMorrisonUp(self.b2, self.invA, self.n)
      self.assertMatrixAlmosteEqual(self.invAb1b1T, invAxxH, 15)
      
      invAxxH = wb.shermanMorrisonUp(self.complexb, self.complexInvA, self.n)
      self.assertMatrixAlmosteEqual(self.complexInvAbbH, invAxxH, 15)
      
   
   def testIterativBuildRinv(self):
      
      M = self.sonardata_n
      L = 24
      Yavg = 5
      d = 0 # diagonal loading as a function of matrix trace is not supported. Use the squared sum of channel data (i.e. the total energy)
      
      invR = wb.iterativeBuildRinv(self.sonardata_ar, M, L, Yavg, d)
      invRa = mynp.dot(invR, self.sonardata_a)
      self.assertMatrixAlmosteEqual(self.sonardata_Ria, invRa, 7)
      
      x = mynp.ones(M, dtype=complex)
      invR = wb.iterativeBuildUpRinv(invR, x, M, L)
      invR = wb.iterativeBuildDownRinv(invR, x, M, L)
      invRa = mynp.dot(invR, self.sonardata_a)
      self.assertMatrixAlmosteEqual(self.sonardata_Ria, invRa, 7) 
   
   
def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestWoodbury)
