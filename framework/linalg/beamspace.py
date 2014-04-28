'''
Created on 20. jan. 2012

@author: jpaasen_adm
'''

import framework.mynumpy as mynp

import cmath as cm
import math as ma

def beamspace(B, x):
   ''' Transform/beamform the given data vector x with the matrix B '''
   ''' This is just a wrapper for a standard matrix-vector multiplication '''
   return mynp.dot(B, x)


def butlerMatrix(m, n, ix = 0):
   ''' Returns the mxn buttler matrix used for beamspace processing '''
   ''' The matrix is equal to the normalized n-point DFT-matrix     '''
   
   ' An optional list argument ix can be specified to select different beams than the first m'

   B = mynp.zeros([m, n], dtype=complex)
   r = range(m)
   
   if ix != 0:
      r = ix

   for i in r:
      for j in range(n):
         B[i, j] = 1/ma.sqrt(n) * cm.exp(-1j*2*cm.pi*i*j/n)
   
   return B
