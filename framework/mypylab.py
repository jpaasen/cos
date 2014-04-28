from matplotlib.pylab import *
import matplotlib.ticker as ticker
#ion, show, \
#                  clim, cm, colorbar, \
#                  draw, \
#                  get_current_fig_manager, grid, \
#                  ioff, ion, \
#                  legend, \
#                  setp, subplot, subplots_adjust, \
#                  xlabel, xlim,  \
#                  ylabel, ylim



import numpy as np

import pylab as pl
from gfuncs import swarning, is_np_array_type

import os
PHDCODE_ROOT      = os.environ['PHDCODE_ROOT']
GFX_PATH          = PHDCODE_ROOT+'/gfx'
FIG_FORMAT        = 'pdf'
#FIG_FORMAT        = 'png'
INTERACTIVE_MODE  = False
AUTO_EXPORT       = False
FIG_LAYOUT        = True

#pl.rcParams['figure.figsize'] = (11,8)

fig_no = 0
def figure(mode='mosaic',increment_fig_no=True):
   global fig_no
   global INTERACTIVE_MODE
   
   screen_res = [1920, 1080]
   
   def make_disp_map(mode, screen_res, fig_width, fig_height, fig_no):
         
      if mode == 'diagonal':
         max_x = int(pl.floor((screen_res[0]-fig_width)/20))
         max_y = int(pl.floor((screen_res[1]-fig_height)/20))
      elif mode == 'mosaic':
         max_x = int(pl.floor(screen_res[0]/fig_width))
         max_y = int(pl.floor(screen_res[1]/fig_height))
         
      disp_map = [] #[(0, 60, fig_width, fig_height)]

      top_offset = 60
      
      for y in range(0,max_y):
         if y == max_y-1:
            mx = max_x-1
         else:
            mx = max_x
         for x in range(0,mx):
            if mode=='mosaic':
               disp_map.append((x*fig_width, top_offset+y*fig_height, fig_width, fig_height))
            elif mode=='diagonal':
               disp_map.append((20*x*y, top_offset+15*y*x, fig_width, fig_height))
               
      return disp_map[np.mod(fig_no, max_x*max_y-1)]
   
   
   fn = pl.figure()
   figman = pl.get_current_fig_manager()
#   fig_geometry = figman.window.geometry().getCoords()
   fig_width  = 200#fig_geometry[2] #0.5*screen_res[0]
   fig_height = 200#fig_geometry[3] #0.7*screen_res[1]
   dm = make_disp_map(mode, screen_res, fig_width, fig_height, fig_no)
   if increment_fig_no:
      fig_no += 1
               
   if INTERACTIVE_MODE:
#      print '%d %d %d %d' %(dm[0], dm[1], dm[2], dm[3]) 
      figman.window.setGeometry(dm[0], dm[1], dm[2], dm[3]) # [0], 85, 800, 600
      return fn

   else:
#      from subprocess import call
#      call(["cp", "%s/waiting.png"%GFX_PATH, "%s/%d.png"%(GFX_PATH,fig_no)])
#      call(["gwenview", "%s/%d.png"%(GFX_PATH,fig_no)])
      return fn

import threading
class SaveMyFig( threading.Thread ):
   def __init__(self, *args, **kwargs):
      threading.Thread.__init__(self)
      self.args = args
      self.kwargs = kwargs
      
   def run ( self ):
      # When savefig() is called without parameters, figures are stored using successive numbers
      if self.args.__len__() == 0:
         filename = "%s/%d.%s"%(GFX_PATH,fig_no,FIG_FORMAT)
         pl.savefig(filename, **self.kwargs)
         
      else:
         pl.savefig(*self.args, **self.kwargs)
   

def savefig(*args, **kwargs):
#   import subprocess as sp
   savefig_thread = SaveMyFig(*args, **kwargs)
   savefig_thread.start()
#   savefig_thread.run( *args, **kwargs )
   


def close(*args):
   global fig_no
   if args.__len__() != 0 and type(args[0]) == str and args[0] == 'all':
      pl.close('all')
      fig_no = 0
   else:
      pl.close(*args)
      fig_no -= 1
  

def plot(*args, **kwargs):
#   from TypeExtensions import Ndarray
   global INTERACTIVE_MODE
   
   if args.__len__() > 1:
      xaxis = args[0]
      yaxis = args[1]
   else:
      xaxis = None
      yaxis = args[0]
      
   
#   def subplot(ax, x, y, **kwargs):
#      if x == None:
#         if isinstance(y, np.Ndarray):
#            try:
#               ax.plot(y.axis, y, **kwargs)
#            except:
#               ax.plot(y, **kwargs)
#            try:
#               ax.set_ylabel(y.axes.name(0))
#            except: pass
#            try:
#               ax.set_xlabel(y.axis.axes.name(0))
#            except: pass
#            try:
#               ax.set_title(y.desc)
#            except: pass
#         else:
#            ax.plot(y, **kwargs)
#      else:
#         ax.plot(x, y, **kwargs)
   
   try:
      fn = kwargs.pop('fn')
      ax = fn.add_axes([0.1,0.1,0.8,0.8])
   except:
      try:
         ax = kwargs.pop('ax')
      except:
         fn = figure()
         ax = fn.add_axes([0.1,0.1,0.8,0.8])
   

   # Recursively plot data in 'y'
   def subplot_y(ax, y, **kwargs):
      if is_np_array_type(y.__class__):
         if y.ndim > 1:
            for i in range(y.shape[0]):
               subplot_y(ax, y[i], **kwargs)
         else:
            ax.plot(y, **kwargs)
      elif isinstance(y, list):
         for i in range(y.__len__()):
            subplot_y(ax, y[i], **kwargs)
      else:
         ax.plot(y, **kwargs)

   # Exactly the same code as above, but with common x-axis
   def subplot_xy(ax, x, y, **kwargs):
      if is_np_array_type(y.__class__):
         if y.ndim > 1:
            for i in range(y.shape[0]):
               subplot_xy(ax, x, y[i], **kwargs)
         else:
            ax.plot(x, y, **kwargs)
      elif isinstance(y, list):
         for i in range(y.__len__()):
            subplot_xy(ax, x, y[i], **kwargs)
      else:
         ax.plot(x, y, **kwargs)


   if xaxis == None:
      subplot_y(ax, yaxis, **kwargs)

   elif isinstance(xaxis, list):
      if isinstance(yaxis, list):
         if xaxis.__len__() == yaxis.__len__():
            for x,y in zip(xaxis,yaxis):
               ax.plot(x, y, **kwargs)
         else:
            swarning('x and y-axis of different lengths. Only plotting y.')
            for y in yaxis:
               ax.plot(y, **kwargs)
      else:
         swarning('x-axis is a list, y-axis is not. Only plotting x[0].')
         ax.plot(xaxis[0], yaxis, **kwargs)
   else:
      subplot_xy(ax, xaxis, yaxis, **kwargs)

      
   if INTERACTIVE_MODE:
      savefig()


      
def imshow(X, cmap=None, norm=None, aspect='auto', interpolation='nearest', \
           alpha=None, vmin=None, vmax=None, origin=None,extent=None, \
           shape=None, filternorm=1, filterrad=4.0, imlim=None, \
           resample=None, url=None, hold=None, **kwargs):
   
   global INTERACTIVE_MODE
   
   pl.imshow(X, cmap, norm, aspect, interpolation, \
             alpha, vmin, vmax, origin, extent, \
             shape, filternorm, filterrad, imlim, \
             resample, url, hold)
   
   if INTERACTIVE_MODE:
      savefig()
   

def ion(*args, **kwargs):
   global INTERACTIVE_MODE
   
   INTERACTIVE_MODE = True
#   pl.ion(*args, **kwargs)