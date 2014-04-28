'''
Created on Aug 19, 2011

@author: me
'''

import mynumpy as np

@profile
def lca(data_sum, Y_AVG ):
   Ny = data_sum.shape[0]
   Nx = data_sum.shape[1]
   Nw = data_sum.shape[2]
   
   # Select the window that yield the least power
   if Y_AVG == 0:
      l = range(0,Ny)
      m = range(0,Nx)
      l,m = np.meshgrid(range(0,Nx),range(0,Ny))
      selected_window = np.abs(data_sum).argmin(2)
      return data_sum[m,l,selected_window]
            
         
   else:
#            img_lca_all_i[i,:,:] =
      # For any center pixel, select the window that yields the lowest variance
      # for the local segment of pixels - i.e. use the same window on all and see
      # who yields the lowest variance.
      
      # This is the same as iterating over all windows and sum up the range
      # interval, and select the window with the lowest sum.
      
      # Equvalently, one may also define the range of y's, iterate of each of
      # these and select the window that yields the lowest sum.
      
      OFFSET    = Y_AVG
#      WIN_SIZE  = 2*OFFSET+1
      
      img_y          = np.zeros((Nw,Ny+2*OFFSET), dtype=complex)
      new_y          = np.zeros((Ny,), dtype=complex)
      new_w          = np.zeros((Ny,), dtype=int)
      new_img        = np.zeros((Ny,Nx), dtype=complex)
      new_win        = np.zeros((Ny,Nx), dtype=int)
      img_y_segment_abs2  = np.zeros((Nw,2*OFFSET), dtype=float)
      img_y_segment_abs2_sum  = np.zeros((Nw,), dtype=float)
      
      for x in range (0,Nx):
         img_y = data_sum[:,x,:].T
         img_y_abs2 = abs(img_y)**2 # Why abs^2?
      
         img_y_segment_abs2 = img_y_abs2[:,0:2*OFFSET+1]
         img_y_segment_abs2_sum = img_y_segment_abs2.sum(1)
         idx = img_y_segment_abs2_sum.argmin(0)
         new_y[0] = img_y[idx,0]
         new_w[0] = idx
         for y in range(1+OFFSET,Ny-OFFSET):
            img_y_segment_abs2_sum = img_y_segment_abs2_sum - img_y_abs2[:,y-OFFSET-1] + img_y_abs2[:,y+OFFSET]
            idx = img_y_segment_abs2_sum.argmin(0)
            new_y[y-OFFSET] = img_y[idx,y-OFFSET]
            new_w[y-OFFSET] = idx
            
         new_img[:,x] = new_y
         new_win[:,x] = new_w

      return new_img, new_win
