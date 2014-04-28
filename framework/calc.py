# coding=<encoding name>

from gfuncs import error, warning, is_np_array_type

from mynumpy import *
import mynumpy as np



from Coordinate import Coordinate
#from System import System

def delay(system=None, X=None):
   
   # If a system is supplied, attempt to find the relevant parameters
   if system != None:
      try:
         X = system.findAttr('X')
      except:
         error('Unable to find the relevant parameters in supplied system. Aborting.')
         pass
   
   if X==None:
      error('One or more of the required parameters are \'None\'. Aborting.')
      return
   
def derive_system_parameters(s):
   
   from Array import Array
   from Coordinate import Coordinate
   
   if s != None:   
      if s.findAttr('p') == None:
         M = s.findAttr('M')
         d = s.findAttr('d')
         
         if (M,d) != (None,None):
            x = np.linspace(-0.5,0.5,M)*M*d
            
            if not 'array' in s.__dict__:
               s.array = Array()
               
            s.array.p = Coordinate(x=x, y=None, z=None,
                                   axes=['M',('dim',['x','y','z'])])

   
def das():  
   pass

def W(p=None, w=None, k=None, fc=100e3, c=1480, N=500):
   
   if p==None:
      error('\'p\' are missing. Aborting.')
      return
   
   if k!=None:
      k_abs = k
   
   elif p!=None and fc!=None and c!=None:
      k_abs = 2*pi*fc/c
      
   else:
      error('Either \'k\', or \'c\' and \'fc\' must be supplied. Aborting.')
      return
   
   

   theta       = np.linspace(-0.5,0.5,N)*pi
   
   k           = Coordinate(r=k_abs, theta=theta, phi=None)
   
   def compute_W(ax,ay,az, w):
      Wx          = dot( ax, w )
      Wy          = dot( ay, w )
      Wz          = dot( az, w )

      Wxyz = vstack((Wx,Wy,Wz)).T
      
      return Wxyz
   
#   pT = p.T.copy()
   ax = exp(1j*outer( k[:,0], p[:,0]))
   ay = exp(1j*outer( k[:,1], p[:,1]))
   az = exp(1j*outer( k[:,2], p[:,2]))
   
   if type(w) == list:
      W = []
      for wi in w:
         W.append(compute_W(ax,ay,az, wi))
   elif is_np_array_type(w.__class__):
      if w.ndim == 1:
         W = []
         W.append( compute_W(ax,ay,az,w) )
      elif w.ndim == 2:
         W1 = compute_W(ax,ay,az,w[0])
         W = np.zeros((w.shape[0],W1.shape[0],W1.shape[1]),dtype=W1.dtype)
         W[0] = W1
         for i in range(1,w.shape[0]):
            W[i] = compute_W(ax,ay,az, w[i])
      elif w.ndim == 3:
         W1 = compute_W(ax,ay,az,w[0,0]) # Not exactly efficient, but shouldn't be noticeable
         W = np.zeros((w.shape[0],w.shape[1],W1.shape[0],W1.shape[1]),dtype=W1.dtype)
         for i in range(0,w.shape[0]):
            for j in range(0,w.shape[1]):
               W[i,j] = compute_W(ax,ay,az, w[i,j])
         
      else:
         print 'EE: Not implemented.'
      
   else:
      W = []
      W.append( compute_W(ax,ay,az,w) )
      
  
   return W
