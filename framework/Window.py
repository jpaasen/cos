#from TypeExtensions import Ndarray

from gfuncs import processArgs

from mynumpy import pi, dot, cos, sin, exp #ones, complex, sin, linspace, exp, pi, dot, angle
import mynumpy as np
#from pylab import plot, subplot, xlabel, ylabel, grid, show, figure, ion, ioff


class Window(np.Ndarray):
   def __new__(self, type='rect', **kwargs):
      from gfuncs import error
      if type == 'rect':
         return self.rect(self,kwargs)
      elif type == 'kaiser':
         return self.kaiser(kwargs)
      else:
         error(self, 'The window type %s is not recognised'%type)
#
#      Configurable.__init__(self)
#      Operable.__init__(self)

class Rect(np.Ndarray):
   def __new__(self, M=10, phi=0, normalised=True):
            
      # Create the window
      if phi == 0:
         win = np.ones( (M,), dtype=None ) / M
      else:
         wc         = np.ones( M, dtype=complex )   # Window coefficients
         m          = np.arange(0,M)   # Create M indeces from 0 to 1
         a          = exp(-1j*2*pi*m*phi)       # Steering vector
         ws         = dot(wc, a)                      # Normalisation factor
         win         = a * wc / ws                # Steered and normalised window
         
      w = np.Ndarray.__new__(self, win)
#                               axes=('M',),
#                               desc = 'Rectangular (phi=%d)'%phi)
#                               desc='Rectangular (phi=%d)'%phi,
#                               shape_desc=('M','1'))
      return w


class Trig(np.Ndarray):
   def __new__(self, M=10, a=0.54, phi=0, normalised=True):
            
      # Create the window
      if phi == 0:
         wc   = a + (1-a)*np.cos(2*pi*np.linspace(-0.5,0.5,M))
         win  = wc / sum(wc)                   # Normalised window
      else:
         n    = np.linspace(-0.5,0.5,M)
         wc   = a + (1-a)*np.cos(2*pi*n)  # Window coefficients
         m    = np.arange(0,M)            # Create M indeces from 0 to 1
         aa   = exp(-1j*2*pi*m*phi)       # Steering vector
         ws   = dot(wc, aa)               # Normalisation factor
         win  = aa * wc / ws              # Steered and normalised window
         
      w = np.Ndarray.__new__(self, win)
#                               axes=('M',),
#                               desc = 'Rectangular (phi=%d)'%phi)
#                               desc='Rectangular (phi=%d)'%phi,
#                               shape_desc=('M','1'))
      return w 
         

class Kaiser(np.Ndarray):
   '''kaiser( M=10, beta=1, phi=0, normalised=True )
   The Kaiser window is a taper formed by using a Bessel function.
   
    Parameters
    ----------
    M          (int)     : Number of points in the output window.
    beta       (float)   : Shape parameter for window.
    phi        (float)   : Steering angle.
    normalised (boolean) : Use normalised window coefficients?
    '''
         
   def __new__(self, M=10, beta=1, phi=0, normalised=True, inverted=False):

      if not inverted:                     
         if phi == 0:
            wc          = np.kaiser(M, beta)           # Window coefficients
            win         = wc / sum(wc)                   # Normalised window
         else:
            wc          = np.kaiser(M, beta)           # Window coefficients
            m           = np.arange(0,M)              # Create M indeces from 0 to 1
            a           = exp(-1j*2*pi*m*phi)     # Steering vector
            ws          = dot(wc, a)                      # Normalisation factor
            win         = a * wc / ws                # Steered and normalised window
      else:
         if phi == 0:
            wc          = 1 / np.kaiser(M, beta)           # Window coefficients
            win         = wc / sum(wc)                   # Normalised window
         else:
            wc          = 1 / np.kaiser(M, beta)           # Window coefficients
            m           = np.arange(0,M)              # Create M indeces from 0 to 1
            a           = exp(-1j*2*pi*m*phi)     # Steering vector
            ws          = dot(wc,a)                      # Normalisation factor
            win         = a * wc / ws                # Steered and normalised window
   
      w = np.Ndarray.__new__(self, win)
#                               axes=('M',),
#                               desc = 'Kaiser (beta=%d, phi=%d)'%(beta,phi))
#                               shape_desc=('M','1'))
      return w

#   def plot(self, **kwargs):
#      
#      # Set some default options
#      opts = {'magnitude':True, 'angle':False, 'grid':True, 'degrees':True}
#      
#      # Add the user-specified options
#      for key,val in kwargs.iteritems():
#         if opts.has_key(key):
#            opts[key] = val
#         else:
#            opts[key] = val
#            print 'WW: Window.plot() - Supplied parameter '+key+' is unknown.'
#      
#      ion()
#      if opts['magnitude'] and opts['angle']:
#         figure()
#         subplot(2,1,1)
#         plot( abs(self.w) )
#         xlabel( 'Channel #' )
#         ylabel( 'Magnitude' )
#         grid( opts['grid'] )
#         
#         subplot(2,1,2)
#         plot( angle(self.w, deg=opts['degrees']) )
#         xlabel( 'Channel #' )
#         if opts['degrees']:
#            ylabel( 'Angle [degrees]' )
#         else:
#            ylabel( 'Angle [radians]' )
#         grid( opts['grid'] )
##         show()
#      
#      elif opts['magnitude']:
#         figure()
#         plot( abs(self.w) )
#         xlabel( 'Channel #' )
#         ylabel( 'Magnitude' )
#         grid( opts['grid'] )
##         show()
#      
#      else:
#         figure()
#         plot( angle(self.w, deg=opts['degrees']) )
#         xlabel( 'Channel #' )
#         if opts['degrees']:
#            ylabel( 'Angle [degrees]' )
#         else:
#            ylabel( 'Angle [radians]' )
#         grid( opts['grid'] )
##         show()
#      ioff()
