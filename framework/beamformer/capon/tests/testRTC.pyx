'''
Created on 18. aug. 2011

@author: jpaasen
'''
import unittest, sys, os
#import testComplexH as TC
import framework.mynumpy as np
cimport numpy as np

#from framework.lib.TestCaseBase import TestCaseBase
from framework.lib.TestCaseBasec import TestCaseBase

np.set_printoptions(precision=8)
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=300)
np.set_printoptions(threshold=10000)

from libcpp cimport bool

from Complex cimport *
from BuildR cimport *
from Solver cimport *
from Capon cimport *

import time


from .. import getCaponC as getCaponC
from .. import getCaponAllC as getCaponAllC
from .. import getCaponAllCPy as getCaponAllCPy
from .. import getCaponCUDA as getCaponCUDA

FLOATING_PRECISION = 32 #64 for double
VERBOSE = False
DEBUG   = False

class TestRTC(TestCaseBase):
 
   def setUp(self):
      pass
                
   def testBuildRGPU(self):
      
      cdef float d      # Diagonal loading
      cdef int   L      # Subarray size
      cdef int   Yavg   # Temporal averaging
      cdef int   M      # Number of channels
      cdef int   Nx     # Number of pixels
      cdef int   Nz
      cdef int   Nb = 0 # Number of beamspace dimensions
      
      cdef int n,i,j,k,xi,p,nn
      cdef float R_trace, diag_load
      
      cdef bool R_on_gpu = False
      cdef bool x_on_gpu = False
      
      cdef cuComplex* sum
      
      cdef np.ndarray[np.complex64_t, ndim=4] R_py
      cdef cuComplex* R
      
      cdef np.ndarray[np.complex64_t, ndim=3] x_py
      cdef cuComplex* x
      
      cdef np.ndarray[np.complex64_t, ndim=4] Rans_full
      cdef np.ndarray[np.complex64_t, ndim=4] Rans_sub
      
      cdef BuildR buildR
      
#      #              M   L   Yavg d  Nx Nz
#      parameters = [(8,  4,  0,   0, 1, 1)
#                   ,(8,  4, 0,    0.5, 1, 1)]

#      parameters = []
#      for M in [2,3,13,32,47,64]:
#         for L in [M/2, M-1, M]:
#         #for L in [2, M/2, M-1, M]:
#            for Ny in [3, 57, 100]:#, 500]:
#               for Yavg in [0,1]:
#                  for Nx in [2, 10, 100]:
#                     #parameters.append((M, L, Yavg, 0.2, Nx, Ny))

      parameters = []            
      for M in [10,32]:
         for L in [M/6, M/4, M/2]:
         #for L in [2, M/2, M-1, M]:
            for Ny in [3, 57]:#, 500]:
               for Yavg in [0,1]:
                  for Nx in [2, 10]:
                     parameters.append((M, L, Yavg, 0.2, Nx, Ny))
      
      #              M   L   Yavg d  Nx Nz
      #parameters = [(6, 6, 1,   0, 1, 3)]
      #parameters = [(32, 2, 1, 0, 10, 57)]

      if VERBOSE:
         print ""
      for p in range(parameters.__len__()):

         if VERBOSE:
            print "Testing M=%d, L=%d, Yavg=%d, d=%2.2f, Nx=%d, Nz=%d"%parameters[p]
         M, L, Yavg, d, Nx, Nz = parameters[p]
         
         R_py = np.zeros((Nx,Nz-2*Yavg,L,L),dtype=np.complex64)
         R = <cuComplex*>R_py.data
         
         tmp = np.random.normal(size=(Nx,Nz,M)) + 1j*np.random.normal(size=(Nx,Nz,M))
         x_py = tmp.astype('complex64')
         x = <cuComplex*>x_py.data

         buildR.getR (
            R,          # buffer holding the resulting covariance matrices
            x,          # buffer holding data vectors      
            d,          # diagonal loading factor
            L,          # number of spatial sublengths
            Yavg,       # number of samples averaged in time
            M,          # number of data elements 
            Nx,         # number of data vectors in azimuth
            Nz,         # number of data vectors in range
            Nb,         # dimension of beamspace
            R_on_gpu,  # true if you want R to be left on the gpu
            x_on_gpu   # true if x is already on the gpu
            )
         
         ###########################
         # Find answer with Python #
         ###########################
         
         t = time.time()
         
         Rans_full = np.zeros((Nx,Nz-2*Yavg,M,M),dtype=np.complex64)
         Rans_sub  = np.zeros((Nx,Nz-2*Yavg,L,L),dtype=np.complex64)
              
         for xi in range(Nx):
            for n in range(Yavg,Nz-Yavg):
               for nn in range(-Yavg,Yavg+1):
                  for i in range(M):
                     for j in range(M):
                        Rans_full[xi,n-Yavg,i,j] = Rans_full[xi,n-Yavg,i,j] \
                                                 + x_py[xi,n+nn,i]*x_py[xi,n+nn,j].conjugate()
         
            # Sum the subarray covariance matrices
            for n in range(Nz-2*Yavg):
               # Iterate over subarrays
               for k in range(M-L+1):
                  # Iterate over rows in R
                  for i in range(L):
                     # Iterate over columns in R
                     for j in range(L):
                        # Sum the products
                        Rans_sub[xi,n,i,j] = Rans_sub[xi,n,i,j] + Rans_full[xi,n,k+i,k+j]
               
               # Compute the trace      
               R_trace = 0.0
               for i in range(L):
                  R_trace = R_trace + Rans_sub[xi,n,i,i].real
               
               # Compute the diagonal loading factor and add it
               diag_load = R_trace * d / L
               for i in range(L):
                  Rans_sub[xi,n,i,i] = Rans_sub[xi,n,i,i] + diag_load
                  
         #if DEBUG:
         #   print "Rans_sub"
         #   print Rans_sub
         
         #   print "Rsub (GPU)"
         #   print R_py
               
         if VERBOSE:
            elapsed = time.time() - t
            print "Time to calculate R in cython: %f"%elapsed
         
#         self.debug(x_py, Rans_sub, R_py)

         # The sliding window approach used on the GPU accumulates floating point errors (especially along the longest path, the diagonal)
         # The error will increase with K and Yavg.
         self.almostEqual(Rans_sub.conjugate().squeeze(), R_py.squeeze(), precision=3, just_report=True, plot_diff=False)
         

   def testSolver(self):

      cdef int  L        = 3
      cdef int  N        = L        # size of each linear system
      cdef int  batch    = 2        # number of linear systems
      cdef bool x_on_gpu = False    # true if x should remain on the gpu
      cdef bool A_on_gpu = False    # true if R is already on the gpu
      cdef bool b_on_gpu = False    # true if b is already on the gpu
      
      cdef np.ndarray[np.complex64_t, ndim=2] x_py = np.zeros((batch,L),dtype=np.complex64)
      cdef cuComplex* x = <cuComplex*>x_py.data
      
      cdef np.ndarray[np.complex64_t, ndim=2] b_py = np.ones((batch,L),dtype=np.complex64)
      cdef cuComplex* b = <cuComplex*>b_py.data

      tmp = np.random.normal(size=(batch,L,L)) + 1j*np.random.normal(size=(batch,L,L))
      #tmp = tmp.astype('complex64').transpose().copy()
      
      #print tmp.shape
      
      for i in range(batch):
         tmp2  = np.random.normal(size=(L,L)) + 1j*np.random.normal(size=(L,L))
         tmp2[np.diag_indices(L)] = tmp2[np.diag_indices(L)] + 10
         #tmp2 = tmp2.astype('complex64').transpose().copy()
         tmp[i,:,:] = tmp2
         
      tmp = tmp.astype('complex64')
      
      cdef np.ndarray[np.complex64_t, ndim=3] A_py = tmp[:]
      cdef cuComplex* A = <cuComplex*>A_py.data
      cdef Solver solver
      
      solver.solve(
         x,             # buffer holding the solutions
         A,             # buffer holding matrices
         b,             # buffer holding the left sides      
         N,             # size of each linear system
         batch,         # number of linear systems
         x_on_gpu,      # true if x should remain on the gpu
         A_on_gpu,      # true if R is already on the gpu
         b_on_gpu       # true if b is already on the gpu
         )

      x_ref = np.zeros((batch,L), dtype=np.complex64)   
      for i in range(batch):   
         x_ref[i,:] = np.linalg.solve(np.squeeze(A_py[i,:,:].transpose()), b_py[i,:])
      
      if VERBOSE:
         print "Solution (CPU):"
         print x_ref
         print "Solution (GPU):"
         print x_py
      
      self.almostEqual(x_ref, x_py, precision=4)#, just_report=True, plot_diff=False)
      
      
   def testCaponGPU(self):
      
      cdef float d = 0
      cdef int   L = 8
      cdef int   Yavg = 0
      cdef int   M = 16
      cdef int   Nx = 1
      cdef int   Nz = 1
      cdef int   Nb = 0
      
      cdef np.ndarray[np.complex64_t, ndim=3] x_py
      cdef np.ndarray[np.complex64_t, ndim=3] w_py
      cdef np.ndarray[np.complex64_t, ndim=2] z_py
      cdef np.ndarray[np.complex64_t, ndim=4] R_py
      
      cdef cuComplex* x
      cdef cuComplex* w
      cdef cuComplex* z
      cdef cuComplex* R
      
      cdef Capon capon
      
      #                N     M   L
      parameters = []            
      for M in [12,32]:
         for L in [M/6, M/4, M/2]:
         #for L in [2, M/2, M-1, M]:
            for Ny in [3, 57]:#, 500]:
               for Yavg in [0,1]:
                  for Nx in [2, 10]:
                     parameters.append((Nx, Ny, M, L, Yavg, 0.2))

      #parameters = [( 7,   10,  2,  2, 0,    0),
      #              ( 4,    6,  7,  6, 1,    0),
      #              ( 1,    7, 11,  5, 0,  0.1),
      #              ( 2,    1,  6,  3, 0, 0.01),
      #              ( 2,    8, 64, 24, 0,    0),
      #              ( 4,    6, 32, 16, 1,    0),
      #              (11,    7, 32, 16, 0,  0.1),
      #              ( 2,    4,  6,  3, 0, 0.01),
                    # Do not run the test below in DEBUG!!!
                    #(79,  832, 64, 24, 0,    0), # testing setup for csound data
                    #(79,  832, 64, 16, 0,    0),
                    #(79,  832, 64, 16, 1,    0),
                    #(60, 1000, 32, 16, 1,    0), # HUGIN example
                    #(70, 1000, 32, 16, 0,  0.1)] # HUGIN example
         
      # Some sane sonar parameters:
#     parameters = [(86, 640, 32, 16, 0, 0.01)] # HUGIN example
      
      #parameters = [(79, 8320, 64, 24, 0,   0), # testing setup for csound data
      #              (60, 1000, 32, 16, 1,   0), # HUGIN example
      #              (70, 1000, 32, 16, 0, 0.1)] # HUGIN example
      #parameters = [(1, 8, 64, 24, 0,   0),
                    #(2, 8, 64, 24, 0,   0), # testing setup for csound data
                    #(4, 6, 32, 16, 1,   0), # HUGIN example
                    #(11, 7, 32, 16, 0, 0.1)] # HUGIN example   
      
      # Some sane sonar parameters:
      #parameters = [(86, 640, 32, 16, 0, 0.01)] # HUGIN example
      
      # Basic testing:
      #parameters = [(2, 4, 6, 3, 0, 0.01)] # HUGIN example
      
      if VERBOSE:
         print ""

      for i in range(parameters.__len__()):
         
         if VERBOSE:
            print "Testing Nx=%d, Nz=%d, M=%d, L=%d, Yavg=%d, d=%2.2f"%parameters[i]

         Nx, Nz, M, L, Yavg, d = parameters[i]

         x_ref = np.random.uniform(size=(Nx,Nz,M)) + 1j*np.random.uniform(size=(Nx,Nz,M))
         x_py  = x_ref.astype('complex64')

         
         R_py = np.zeros((Nx,Nz-2*Yavg,L,L),dtype=np.complex64)
         z_py = np.zeros((Nx,Nz-2*Yavg,   ),dtype=np.complex64)
         w_py = np.zeros((Nx,Nz-2*Yavg,L  ),dtype=np.complex64)

         
         x = <cuComplex*>x_py.data
         R = <cuComplex*>R_py.data
         z = <cuComplex*>z_py.data
         w = <cuComplex*>w_py.data

         e = capon.getCapon(
            z,                # output amplitude per pixel
            w,                # output weights per pixel
            R,                # buffer holding the resulting covariance matrices
            x,                # buffer holding data vectors      
            d,                # diagonal loading factor
            L,                # number of spatial sublengths
            Yavg,             # number of samples averaged in time
            M,                # number of data elements 
            Nx,               # number of data vectors in azimuth
            Nz,               # number of data vectors in range
            Nb                # dimension of beamspace
            )
         
         if( e != 0 ):
            print "Cuda error (Capon.cpp): Return code %d"%e
            self.assertTrue(False)

         #print ""
         #print "Weights GPU"
         #print w_py
         
         #print "Amplitude GPU"
         #print z_py

         
         #######################
         # COMPARE WITH CAPONC #
         #######################
         
         V = np.array([])
         
         res = getCaponC.getCaponCPy(
            x_ref,
            d,
            L,
            Yavg,
            V,
            False,
            False)
      
         imAmplitude = res[0]
         imPower     = res[1]
         weight      = res[2]
         
         #if DEBUG:
         #   print "Weights"
         #   print weight
         #   print "Weights shape"
         #   print weight.shape
         # 
         #   print "Amplitude"
         #   print imAmplitude
         #   print "Amplitude shape"
         #   print imAmplitude.shape
         
         self.almostEqual(weight.squeeze(), w_py.squeeze(), precision=4, just_report=True, plot_diff=False)
         self.almostEqual(imAmplitude.squeeze(), z_py.squeeze(), precision=4, just_report=True, plot_diff=False)

   def testCaponBeamspace(self):
      
     
      cdef np.ndarray[np.complex128_t, ndim=3] w_py
      cdef np.ndarray[np.complex128_t, ndim=2] z_py
      cdef np.ndarray[np.complex128_t, ndim=4] R_py
      
      cdef cuComplex* x
      cdef cuComplex* w
      cdef cuComplex* z
      cdef cuComplex* R
      
      cdef Capon capon
      
      #VERBOSE = True
           
      # Generate test set:
      parameters = []
         
      for M in [12, 32, 64]:#,32]:        
         for L in [M/2]:#[M/6, M/4, M/2]:
            for Nb in [2, L/2, L/2+1]: #Nb==L gives error
               
               # make DFT matrix aka Butler matrix
               V = np.zeros((L,L), dtype=np.complex128)
               for b in range(L):
                  for l in range(L):
                     V[b,l] = np.e**(-2j*np.pi*b*l/L) / np.sqrt(L)
                     
               if Nb < L:
                  V = np.delete(V, np.s_[Nb/2+1:L-Nb/2], axis=0) # remove beams 
                  
               for Nz in [3, 23]:#, 500]:
                  for Yavg in [0,1]:
                     for Nx in [2, 10]:
                        parameters.append((Nx, Nz, M, L, Yavg, 0.2, Nb, V))
      
      doFBAvg = False
      verbose = False
      
      if VERBOSE:
         print ""

      for i in range(parameters.__len__()):
         
         #if VERBOSE:
         #   print "Testing Nx=%d, Nz=%d, M=%d, L=%d, Yavg=%d, d=%2.2f, Nb=%d"%parameters[i]

         Nx, Nz, M, L, Yavg, d, Nb, V = parameters[i]

         x_ref = np.random.uniform(size=(Nx,Nz,M)) + 1j*np.random.uniform(size=(Nx,Nz,M))
         R_py = np.zeros((Nx,Nz,L,L),dtype=np.complex128)
         z_py = np.zeros((Nx,Nz,),dtype=np.complex128)
         w_py = np.zeros((Nx,Nz,L),dtype=np.complex128)
               

         ##########################################
         # COMPARE GPU-Implementation WITH CAPONC #
         ##########################################         
## Does not support Beamspace yet ###########         
         res = getCaponC.getCaponCPy(
            x_ref,
            d,
            L,
            Yavg,
            V,
            doFBAvg,
            verbose)
#############################################
         if VERBOSE:
            print "getCaponCPy finished"
         
         getCaponAllCPy.BEAMSPACE = True
         getCaponAllCPy.DIAGONAL_LOADING_IN_BEAMSPACE = True
         res2 = getCaponAllCPy.getCaponCPy(
            x_ref,
            d,
            L,
            Yavg,
            V,
            doFBAvg,
            verbose)
         
         if VERBOSE:
            print "getCaponAllCPy finished"
         
         res3 = getCaponCUDA.getCaponCUDAPy(
            x_ref,
            d,
            L,
            Yavg,
            V,
            doFBAvg,
            verbose)
         
         if VERBOSE:
            print "getCaponCUDAPy finished"
         
         z1,z2,z3          = res[0], res2[0], res3[0]
         zPow1,zPow2,zPow3 = res[1], res2[1], res3[1]
         w1,w2,w3          = res[2], res2[2], res3[2]
         
         #print "z"
         #print z1
         #print z2
         #print z3
         #print "w"
         #print w1
         #print w2
         #print w3
         
         #print w2[0,0,:].shape
         #print np.dot(V.conjugate().T, w2[0,0,:].T)

         self.almostEqual(z2, z3, precision=5, just_report=True, plot_diff=False)

   def ttestCaponWithMatlab(self):
      
      from mlabwrap import mlab
      
      cdef float d = 0
      cdef int   L = 8
      cdef int   Yavg = 0
      cdef int   M = 16
      cdef int   Nx = 1
      cdef int   Nz = 1
      cdef int   Nb = 0
      
      cdef np.ndarray[np.complex128_t, ndim=3] w_py
      cdef np.ndarray[np.complex128_t, ndim=2] z_py
      cdef np.ndarray[np.complex128_t, ndim=4] R_py
      
      cdef cuComplex* x
      cdef cuComplex* w
      cdef cuComplex* z
      cdef cuComplex* R
      
      cdef Capon capon
      
      if VERBOSE:
         print os.environ['PHDCODE_ROOT']+'/framework/beamformer/capon'
      mlab.addpath(os.environ['PHDCODE_ROOT']+'/framework/beamformer/capon')
      #                N     M   L

      #parameters = [(79,  832, 64, 24, 0,   0), # testing setup for csound data
      #              (60, 1000, 32, 16, 1,   0), # HUGIN example
      #              (70, 1000, 32, 16, 0, 0.1)] # HUGIN example
#      parameters = [(1, 8320, 64, 24, 0,   0), # testing setup for csound data
#                    #(1, 6000, 32, 16, 1,   0), # HUGIN example
#                    (1, 7000, 32, 16, 0, 0.1)] # HUGIN example
      #parameters = [(5, 64, 24), # testing setup for csound data
      #              (6, 32, 16), # HUGIN example
      #              (7, 32, 16)] # HUGIN example
      
      # Some sane sonar parameters:
#      parameters = [(86, 640, 32, 16, 0, 0.01)] # HUGIN example
      
      # Basic testing:
#      parameters = [(2, 4, 6, 3, 0, 0.01)] # HUGIN example
      parameters = [(2, 2, 4, 2, 0, 0.01)] # HUGIN example
      
      #parameters = [(5, 5, 4), # testing setup for csound data
                   #(6, 3, 3), # HUGIN example
                   #(7, 4, 2)] # HUGIN example
      
      if VERBOSE:
         print ""

      for i in range(parameters.__len__()):
         
         if VERBOSE:
            print "Testing Nx=%d, Nz=%d, M=%d, L=%d, Yavg=%d, d=%2.2f"%parameters[i]

         Nx, Nz, M, L, Yavg, d = parameters[i]

         x_ref = np.random.uniform(size=(Nx,Nz,M)) + 1j*np.random.uniform(size=(Nx,Nz,M))
         R_py = np.zeros((Nx,Nz,L,L),dtype=np.complex128)
         z_py = np.zeros((Nx,Nz,),dtype=np.complex128)
         w_py = np.zeros((Nx,Nz,L),dtype=np.complex128)
         
         mlab.getCaponMatlab_ref = mlab._make_mlab_command('getCaponMatlab_ref', nout=2, doc=mlab.help('getCaponMatlab_ref'))
         mlab.getCaponMatlab_solve = mlab._make_mlab_command('getCaponMatlab_solve', nout=2, doc=mlab.help('getCaponMatlab_solve'))
#         z_py, power_py = mlab.getCaponMatlab(x_ref, [0], [0], d, L, Yavg, [], False, False)
         A, B = mlab.getCaponMatlab_solve(x_ref, [0], [0], d, L, Yavg, [], False, False)
         
         if VERBOSE:
            print ""
            print A

         #######################
         # COMPARE WITH CAPONC #
         #######################
         
         V = np.array([])
         
         res = getCaponC.getCaponCPy(
            x_ref,
            d,
            L,
            Yavg,
            V,
            False,
            False)
         
         imAmplitude = res[0]
         imPower     = res[1]
         weight      = res[2]
         
         if VERBOSE:
            print ""
            print imAmplitude
         
            #print "Weights"
            #print weight
            #print weight.shape
         
            #print "Amplitude"
            #print imAmplitude
            #print imAmplitude.shape
         
            print A.shape
            print imAmplitude.shape
         
#         self.almostEqual(weight.squeeze(), w_py.squeeze(), precision=4, just_report=True, plot_diff=False)
         self.almostEqual(imAmplitude.squeeze(), A.squeeze(), precision=7, just_report=True, plot_diff=False)
         
   def ttestCaponWithMex(self):
      
      from mlabwrap import mlab
      
      cdef float d = 0
      cdef int   L = 8
      cdef int   Yavg = 0
      cdef int   M = 16
      cdef int   Nx = 1
      cdef int   Nz = 1
      cdef int   Nb = 0
      
      cdef np.ndarray[np.complex128_t, ndim=3] w_py
      cdef np.ndarray[np.complex128_t, ndim=2] z_py
      cdef np.ndarray[np.complex128_t, ndim=4] R_py
      
      cdef cuComplex* x
      cdef cuComplex* w
      cdef cuComplex* z
      cdef cuComplex* R
      
      cdef Capon capon
      
      if VERBOSE:
         print os.environ['PHDCODE_BUILD']+'/framework/beamformer/capon'
      mlab.addpath(os.environ['PHDCODE_BUILD']+'/framework/beamformer/capon')
      #                N     M   L

      #parameters = [(79,  832, 64, 24, 0,   0), # testing setup for csound data
      #              (60, 1000, 32, 16, 1,   0), # HUGIN example
      #              (70, 1000, 32, 16, 0, 0.1)] # HUGIN example
#      parameters = [(1, 8320, 64, 24, 0,   0), # testing setup for csound data
#                    #(1, 6000, 32, 16, 1,   0), # HUGIN example
#                    (1, 7000, 32, 16, 0, 0.1)] # HUGIN example
      #parameters = [(5, 64, 24), # testing setup for csound data
      #              (6, 32, 16), # HUGIN example
      #              (7, 32, 16)] # HUGIN example
      
      # Some sane sonar parameters:
#      parameters = [(86, 640, 32, 16, 0, 0.01)] # HUGIN example
      
      # Basic testing:
      parameters = [(2, 4, 6, 3, 0, 0.01)] # HUGIN example
#      parameters = [(4, 3, 2, 2, 0, 0.01)] # HUGIN example
      
      #parameters = [(5, 5, 4), # testing setup for csound data
                   #(6, 3, 3), # HUGIN example
                   #(7, 4, 2)] # HUGIN example
      
#      print ""

      for i in range(parameters.__len__()):
         
         if VERBOSE:
            print "Testing Nx=%d, Nz=%d, M=%d, L=%d, Yavg=%d, d=%2.2f"%parameters[i]

         Nx, Nz, M, L, Yavg, d = parameters[i]

         x_ref = np.random.uniform(size=(Nx,Nz,M)) + 1j*np.random.uniform(size=(Nx,Nz,M))
#         x_ref = np.arange(2*3*4).reshape(4,3,2)+1j*np.arange(4*3*2).reshape(4,3,2)
         R_py = np.zeros((Nx,Nz,L,L),dtype=np.complex128)
         z_py = np.zeros((Nx,Nz),dtype=np.complex128)
         w_py = np.zeros((Nx,Nz,L),dtype=np.complex128)
         
         mlab.getCaponMatlab_ref = mlab._make_mlab_command('getCaponMatlab_ref', nout=2, doc=mlab.help('getCaponMatlab_ref'))
         mlab.getCaponMatlab_solve = mlab._make_mlab_command('getCaponMatlab_solve', nout=2, doc=mlab.help('getCaponMatlab_solve'))
         mlab.getCaponMex = mlab._make_mlab_command('getCaponMex', nout=3, doc=mlab.help('getCaponMex'))
#         z_py, power_py = mlab.getCaponMatlab(x_ref, [0], [0], d, L, Yavg, [], False, False)
         A, B, C = mlab.getCaponMex(x_ref, d, L, Yavg, [1+1j], False, False)
         
         if VERBOSE:
            print ""
            
            print "z in testRTC (from getCaponMex)"
            print A

         #######################
         # COMPARE WITH CAPONC #
         #######################
         
         V = np.array([])
         
         res = getCaponC.getCaponCPy(
            x_ref,
            d,
            L,
            Yavg,
            V,
            False,
            False)
      
         imAmplitude = res[0]
         imPower     = res[1]
         weight      = res[2]
         
         if VERBOSE:
            print "z in testRTC (direct cython call)"
            print imAmplitude
         
         #print "Weights"
         #print weight
         #print weight.shape
         
         #print "Amplitude"
         #print imAmplitude
         #print imAmplitude.shape
         
            print A.shape
            print imAmplitude.shape
         
#         self.almostEqual(weight.squeeze(), w_py.squeeze(), precision=4, just_report=True, plot_diff=False)
         self.almostEqual(imAmplitude.squeeze(), A.squeeze(), precision=6, just_report=True, plot_diff=False)

def suite():
   return unittest.TestLoader().loadTestsFromTestCase(TestRTC)
