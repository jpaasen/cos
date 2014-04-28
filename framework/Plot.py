from Configurable import Configurable

from gfuncs import processArgs

from numpy import angle, pi

from pylab import plot, subplot, xlabel, ylabel, grid, show, figure, ion, ioff

import pylab as pl

class Plot:
   def __init__(self, system):
      self.fn = 0
      self.system = system



   def W(self, hold=False):
      s     = self.system
      theta = s.data.W.theta
      W     = s.data.W.W
            
      if not hold:
         self.newFig()
         
      theta_deg = 180*theta/pi
         
      pl.plot( theta_deg, abs(W[:,0]) )
      pl.title( 'Aperture Smoothing Function' )
      pl.xlabel( r'\theta' )
      pl.ylabel( r'$|$W(k$_\theta$)$|$' )
      pl.xlim( -90, 90 )
      pl.grid(True)
      pl.draw()

   def plot(self, **kwargs):
      
      # Set some default options
      opts = {'magnitude':True, 'angle':False, 'grid':True, 'degrees':True}
      
      # Add the user-specified options
      for key,val in kwargs.iteritems():
         if opts.has_key(key):
            opts[key] = val
         else:
            opts[key] = val
            print 'WW: Window.plot() - Supplied parameter '+key+' is unknown.'
      
      ion()
      if opts['magnitude'] and opts['angle']:
         figure()
         subplot(2,1,1)
         plot( abs(self.w) )
         xlabel( 'Channel #' )
         ylabel( 'Magnitude' )
         grid( opts['grid'] )
         
         subplot(2,1,2)
         plot( angle(self.w, deg=opts['degrees']) )
         xlabel( 'Channel #' )
         if opts['degrees']:
            ylabel( 'Angle [degrees]' )
         else:
            ylabel( 'Angle [radians]' )
         grid( opts['grid'] )
#         show()
      
      elif opts['magnitude']:
         figure()
         plot( abs(self.w) )
         xlabel( 'Channel #' )
         ylabel( 'Magnitude' )
         grid( opts['grid'] )
#         show()
      
      else:
         figure()
         plot( angle(self.w, deg=opts['degrees']) )
         xlabel( 'Channel #' )
         if opts['degrees']:
            ylabel( 'Angle [degrees]' )
         else:
            ylabel( 'Angle [radians]' )
         grid( opts['grid'] )
#         show()
      ioff()
