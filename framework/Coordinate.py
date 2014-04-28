#from Configurable import Configurable
#from Operable import Operable
#from TypeExtensions import Ndarray

from mynumpy import vstack, ndim, zeros, sqrt, ndarray, array, dot, ones, sin, cos
import mynumpy as np
from gfuncs import is_array, is_scalar

def cartesian(x, y=None, z=None):
   
   max_dim = 1
   for val in (x, y, z):
      if is_array(type(val)):
         if val.size > max_dim:
            max_dim = val.size
   
   if max_dim != 1:
      if is_scalar(type(x)) or (is_array(type(x)) and x.shape == (1,)):
         x = ones(max_dim)*x
      if is_scalar(type(y)) or (is_array(type(y)) and y.shape == (1,)):
         y = ones(max_dim)*y
      if is_scalar(type(z)) or (is_array(type(z)) and z.shape == (1,)):
         z = ones(max_dim)*z
         
   if type(x) == type(None):
      x = zeros(max_dim)
   if type(y) == type(None):
      y = zeros(max_dim)
   if type(z) == type(None):
      z = zeros(max_dim)
      
   if x.ndim == y.ndim == z.ndim == 1:
      if x.shape == y.shape == z.shape:
         return vstack((x,y,z)).T

   print 'EE: CoordinateSystem.cartesian() - This should not happen!'

   
def spherical(r, theta=None, phi=None):
   
#   def repetiveStuff():
#      self.type   = 'spherical'
#      self.shape  = self.p.shape
   
   max_dim = 1
   for val in (r, theta, phi):
      if is_array(type(val)):
         if val.size > max_dim:
            max_dim = val.size
   
   if max_dim != 1:
      if is_scalar(type(r)) or (is_array(type(r)) and r.shape == (1,)):
         r = ones(max_dim)*r
      if is_scalar(type(theta)) or (is_array(type(theta)) and theta.shape == (1,)):
         theta = ones(max_dim)*theta
      if is_scalar(type(phi)) or (is_array(type(phi)) and phi.shape == (1,)):
         phi = ones(max_dim)*phi
         
   if type(r) == type(None):
      r = zeros(max_dim)
   if type(theta) == type(None):
      theta = zeros(max_dim)
   if type(phi) == type(None):
      phi = zeros(max_dim)
      
   if r.ndim == theta.ndim == phi.ndim == 1:
      if r.shape == theta.shape == phi.shape:
         p = vstack((r,theta,phi)).T

#            repetiveStuff()
         return p
   
   print 'EE: CoordinateSystem.spherical() - This should not happen!'


def s2c(p):
#   if self.type != 'spherical':
#      print 'EE: Huh? The coordinates are not spherical...'
#      return
   
   if p.shape[1] == 1:
      print 'EE: Can\'t do much with a single coordinate...'
      return
   
   elif p.shape[1] == 2:
   
      r     = p[:,0]
      theta = p[:,1]
#         phi   = zeros(self.p[:,3]
      
#         rho = 
      z   = r*cos(theta)
      y   = zeros(z.size)
      x   = r*sin(theta)
            
      p[:] = vstack((x,y,z)).T
   
   elif p.shape[1]  == 3:
      r     = p[:,0]
      theta = p[:,1]
      phi   = p[:,2]

      rho = r*sin(theta)
      z   = r*cos(theta)
      y   = rho*sin(phi)
      x   = rho*cos(phi)
      
      p = Coordinate(x=x,y=y,z=z)
      return p
      

class Coordinate(np.Ndarray):
   """The type of coordinate system is simply chosen, and will be inherited by the classes
   requiring this information"""
   
#   azimuth_ref = 'y-axis' 
     
#   def __new__(cls, **kwargs):
#      pass
#      cls.p = Ndarray(self, cls.p, shape_desc=('x','y','z'))
#      return cls
     
   def __new__(cls, **kwargs):
#      self.p = None
#      Operable.__init__(self, 'p')
#      Configurable.__init__(self)
      
      if len(kwargs) != 0:
         if kwargs.has_key('x') and kwargs.has_key('y') and kwargs.has_key('z'):
            coords = cartesian(kwargs.pop('x'), kwargs.pop('y'), kwargs.pop('z'))
            self = np.Ndarray.__new__(Coordinate, coords,
                                   **kwargs)
            self.desc = 'Cartesian'
            self.__init__()
            return self
         
         elif kwargs.has_key('r') and kwargs.has_key('theta') and kwargs.has_key('phi'):
            coords = spherical(kwargs.pop('r'), kwargs.pop('theta'), kwargs.pop('phi'))
#            coords = s2c(coords)
            self = np.Ndarray.__new__(Coordinate, s2c(coords),
                                   **kwargs)
            self.desc = 'Cartesian'
            self.__init__()
            return self
         
         else:
            print 'EE: Coordinate.__init__() - This should not happen!'
      
   def __init__(self, **kwargs):
      self.setAxesUnits()
      self.setAxesDesc()
      
   def getData(self):
      return self.p
   
   def setAxesUnits(self, x='m',y='m',z='m',r='m',theta='rad',phi='rad'):
      from gfuncs import error
      
      if self.desc == 'Cartesian':
         self.units = (x,y,z)
      elif self.desc == 'Spherical':
         self.units = (r,theta,phi)
      else:
         error(self, 'Unknown coordinate system.')
      
   def setAxesDesc(self, template='',
                         x='',y='',z='',
                         r='',theta='',phi=''):
      
      from string import lower
      from gfuncs import error
      
      if lower(template)=='ISO 31-11':
         r,theta,phi = 'Radial distance', 'Inclination angle', 'Azimuthal angle'
      
      elif lower(template)=='hugin':
         x,y,z = 'Along track', 'Cross track', 'Depth'
         r,theta,phi = 'Range', 'Look angle', 'Slant angle'
         
      if self.desc == 'Cartesian':
         self.units = (x,y,z)
      elif self.desc == 'Spherical':
         self.units = (r,theta,phi)
      else:
         error(self, 'Unknown coordinate system.')
   


#   def shape(self):
#      return self.p.shape
   
#   def c2s(self):
#      if self.type == 'cartesian':
#         return sqrt(sum(self.p**2),1)[:,None]
#      elif self.type == 'spherical':
#         return self.p
#      
#   def angle(self, degrees=False):
#      if self.type == 'cartesian':
#         if degrees
#         return sqrt(sum(self.p**2),1)[:,None]
#      elif self.type == 'spherical':
#         return self.p
   
