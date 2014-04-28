#cython.profile(False)
#cython.cdivision(True)
#cython: wraparound=False
#cython: boundscheck=False

import numpy as np
cimport numpy as np


cdef inline mode_1d(data_):

   shape = data_.shape
   
   cdef np.ndarray[np.float64_t, ndim=1] data = data_
   cdef int max, idx, i

   unique_vals = np.unique(data)
   data_freq   = []
   mode = 0
   
   for i in unique_vals:
      data_copy = np.zeros(shape)
      data_copy[data==i] = 1
      data_freq.append(sum(data_copy))
   
   max = 0
   idx = 0
   for i in data_freq:
      if i > max:
         max = i
         mode = unique_vals[idx]
      idx += 1
       
   return mode

cdef mode_(data,int dim=0):
    
   cdef int i,j, shape_A, shape_B
    
   cdef int dims  = np.size(data.shape)
   cdef np.ndarray[np.float64_t, ndim=1] modes1
   cdef np.ndarray[np.float64_t, ndim=2] modes2
   
#    cdef np.ndarray[np.int_t, ndim=1] loc   = np.zeros(Ndim,dtype=np.int)
   
   if dims == 1:
      return mode_1d(data)
   elif dims == 2:
      if dim == 0:
         modes1 = np.zeros(data.shape[1])
         for i in np.range(data.shape[1]):
            modes1[i] = mode_1d(data[:,i])
      else:
         modes1 = np.zeros(data.shape[0])
         for i in np.arange(data.shape[0]):
            modes1[i] = mode_1d(data[i,:])
      return modes1
   elif dims == 3:
      if dim == 0:
         shape_A = data.shape[1]
         shape_B = data.shape[2]
         modes2 = np.zeros((shape_A,shape_B))
         for i in np.arange(shape_A):
            for j in np.arange(shape_B):
               modes2[i,j] = mode_1d(data[:,i,j])
      elif dim == 1:
         shape_A = data.shape[0][:]
         shape_B = data.shape[2][:]
         modes2 = np.zeros((shape_A,shape_B))
         for i in np.arange(shape_A):
            for j in np.arange(shape_B):
               modes2[i,j] = mode_1d(data[i,:,j])
      else:
         shape_A = data.shape[0]
         shape_B = data.shape[1]
         modes2 = np.zeros((shape_A,shape_B))
         for i in np.arange(shape_A):
            for j in np.arange(shape_B):
               modes2[i,j] = mode_1d(data[i,j,:])
      return modes2

