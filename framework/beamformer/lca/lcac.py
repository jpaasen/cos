#cython: profile=False
#cython: cdivision=True
#cython: wraparound=False
#cython: boundscheck=False
# filename: lcacx.pyx

import numpy as np

#################
## Cython mode ##
#################

cimport numpy as np
cimport cython

CTYPE = np.complex
FTYPE = np.float

ctypedef np.complex_t CTYPE_t
ctypedef double complex DCTYPE_t
ctypedef np.float_t FTYPE_t
ctypedef np.int_t ITYPE_t


def lca(np.ndarray[DCTYPE_t, ndim=3] Z_in   not None,
        np.ndarray[FTYPE_t,  ndim=2] wrf_in not None):
   
   cdef int                          Ny
   cdef int                          Nx
   cdef int                          Nw
   cdef np.ndarray[DCTYPE_t, ndim=3] Z
        
   cdef int                          Y_AVG
   cdef int                          Y_MID
   cdef int                          X_AVG
   cdef int                          X_MID
   cdef np.ndarray[FTYPE_t,  ndim=2] wrf
   
   cdef np.ndarray[DCTYPE_t, ndim=1] new_y
   cdef np.ndarray[ITYPE_t,  ndim=1] new_w
   cdef np.ndarray[FTYPE_t,  ndim=2] img_y_segment_abs
   cdef np.ndarray[FTYPE_t,  ndim=1] img_y_segment_abs_sum
   
   cdef FTYPE_t                      w_sum_min
   cdef np.ndarray[FTYPE_t,  ndim=1] w_sum
   
   cdef np.ndarray[DCTYPE_t, ndim=2] img_y
   cdef np.ndarray[FTYPE_t,  ndim=2] img_y_abs
   cdef np.ndarray[DCTYPE_t, ndim=2] new_img
   cdef np.ndarray[ITYPE_t,  ndim=2] new_win
   
   cdef int x,y,w,xx,yy
   
   # For XY-averaging:
   cdef np.ndarray[FTYPE_t,  ndim=3] img_abs

#################
## Python mode ##
#################

#def lca(Z_in, wrf_in):
   
## End ##

   
   Ny = Z_in.shape[0]
   Nx = Z_in.shape[1]
   Nw = Z_in.shape[2]
   Z  = Z_in
   
   if wrf_in.ndim != 2:
      print "You must specify a 2D 'wrf' filter. A (1,1) filter with the value 1 equals no filtering."
      return
   
   Y_AVG = wrf_in.shape[0]
   Y_MID = (Y_AVG-1)/2
   X_AVG = wrf_in.shape[1]
   X_MID = (X_AVG-1)/2 
   wrf   = wrf_in
   
   # The filter must have odd number of rows/columns
   if Y_AVG%2==0 or X_AVG%2==0:
      print "The 'wrf' filter must have odd number of rows/columns."
      return
   
   # If only range-values are supplied, and these are all 1's, then set the
   # 'Y_ONLY_ONES' flag which will be used later to simplify computations.
   if X_AVG == 1:
      Y_ONLY_ONES = True
      for y in range(Y_AVG):
         if wrf[y,0] != 1:
            Y_ONLY_ONES = False
   else:
      Y_ONLY_ONES = False
   
   
   new_y                   = np.zeros((Ny,),           dtype=complex)
   new_w                   = np.zeros((Ny,),           dtype=int)
   img_y_segment_abs       = np.zeros((Nw,2*Y_AVG),    dtype=float)
   img_y_segment_abs_sum   = np.zeros((Nw,),           dtype=float)

   w_sum                   = np.zeros((Nw,),           dtype=float)
   w_sum_min               = 0
   
   img_y                   = np.zeros((Nw,Ny+2*Y_MID), dtype=complex)
   img_y_abs               = np.zeros((Nw,Ny+2*Y_MID), dtype=float)
   new_img                 = np.zeros((Ny,Nx),         dtype=complex)
   new_win                 = np.zeros((Ny,Nx),         dtype=int)
   
   # For XY-averaging:
   img_abs                 = np.zeros((Nw,Ny,Nx),      dtype=float)

   
   # Select the window that yield the least power
   if Y_AVG == 1 and X_AVG == 1 and wrf[0,0]:
      l = range(0,Ny)
      m = range(0,Nx)
      l,m = np.meshgrid(range(0,Nx),range(0,Ny))
      selected_window = np.abs(Z).argmin(2)
      return Z[m,l,selected_window], selected_window
   
   ###############################
   # Perform 'window averaging'. #
   ###############################
   # A way to make the beamformer estimate a pixel more accurately, the beamformer output
   # may be computed for a 'window' of pixels around the one we wish to image, and the window
   # that gave the overall lowest output power may be applied to the center pixel.
   #
   # It is common that the averaging window is comprised of only ones, and no averaging is
   # required in azimuth. That mode is handled first, and will be less computationally
   # intensive than the next 'else if' which handles arbitrary averaging windows.  
   elif Y_ONLY_ONES:
      
      # Iterate over all azimuth coordinates
      for x in range (Nx):
         
         # Compute the power of each range pixel (square them)
         for y in range(Ny):
            for w in range(Nw):
               img_y[w,y] = Z[y,x,w]
               img_y_abs[w,y] = img_y[w,y].real**2 + img_y[w,y].imag**2
         
         # Compute the beamformer output for the first y-segment
         for w in range(Nw):
            w_sum_min = 0
            w_sum[w] = 0
            for y in range(Y_AVG):
               w_sum[w] += img_y_abs[w,y]
               
            # Select the window that yielded the minimum output power of the beamformer
            if w_sum[w] < w_sum_min or w==0:
                  w_sum_min = w_sum[w]
                  new_win[Y_AVG,x] = w
                  new_img[Y_AVG,x] = img_y[w,Y_AVG]
            
         # Select a range segment:
         for y in range(1+Y_MID,Ny-Y_MID):
            # Compute the beamformer output for each of the windows
            for w in range(Nw):
               w_sum[w] += img_y_abs[w,y+Y_MID] - img_y_abs[w,y-Y_MID-1]
                  
               # Select the window that yielded the minimum output power of the beamformer
               if w_sum[w] < w_sum_min or w==0:
                  w_sum_min = w_sum[w]
                  new_win[y,x] = w
                  new_img[y,x] = img_y[w,y]
         
      return new_img, new_win
                  
   # Handle arbitrary window functions:
   elif Y_AVG != 0 and X_AVG != 0:

      # Compute the image absolute value
      for y in range(Ny):
         for x in range(Nx):
            for w in range(Nw):
               img_abs[w,y,x] = Z[y,x,w].real**2 + Z[y,x,w].imag**2
               
#      for w in range(Nw):
#         img_abs[w,:,:] = Z[:,:,w]**2
                              
               
      # Select a range segment:
      for y in range(Y_MID,Ny-Y_MID):
         # Select an azimuth segment:
         for x in range(X_MID,Nx-X_MID):
            # Compute the accumulated beamformer output for each of the windows
            for w in range(Nw):
               w_sum[w] = 0
               for yy in range(2*Y_MID+1):
                  for xx in range(2*X_MID+1):
                     w_sum[w] += img_abs[w,y+yy-Y_MID,x+xx-X_MID]*wrf[yy,xx]

                            
               # Select the window that yielded the minimum output power of the beamformer
               if w_sum[w] < w_sum_min or w==0:
                  w_sum_min = w_sum[w]
                  new_win[y,x] = w
                  new_img[y,x] = Z[y,x,w]

      return new_img, new_win
