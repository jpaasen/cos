#from Operable import Operable
import tables
import os
from Configurable import Configurable

class Data(Configurable):
   '''Results computed by the Calc class.'''
   pass

#   def __init__(self, key, val=None):
#         self.__class__.__name__ = key
#         Operable.__init__(self,key,val)

#   def __init__(self, *args):
#      pass
#      self.W         = None                  # Array beampattern
#      self.Xd         = None                  # Delayed RX data                     [M,N]
#      self.Xdw       = None                  # Delayed and weighted RX data [M,N]
#      self.t          = None                  # Time passed from TX to RX (using ping center as reference)

#   def configure(self, conf): 
#      Operable.configure(self, conf)
#
#
#   def save(self, system):
#      """Regarding compression: http://tukaani.org/lzma/benchmarks.html
#      Best compression to decompression time ratio: zlib at level 5
#      Best speed: lzo at level 9"""
#
#      print 'Saving configuration...'
#      CONF_FILE="data/configuration.lib"
#      
#      conf_file_exists = os.path.exists(CONF_FILE)
#      
#      # Open the file
#      h5 = tables.openFile(CONF_FILE, mode = "w")
#      
#      # Iterate through all system classes
#      for i in system.__dict__.itervalues():
#
#         # If the config is new, add a group for the class
#         if ~conf_file_exists:
#            group = h5.createGroup( '/', i.__class__.__name__ )
#
#         # 
#         class_desc = []
#         for var,val in i.__dict__.itervalues():
#            if type(var) == NoneType:
#               pass
#            else:
#               pass
#      
##         for i in sys_classes_names:
##            if ~conf_file_exists:
#            
#            
##               h5.createTable( group,   )
#      # Open the file
#      h5 = tables.openFile(CONF_FILE, mode = "w")
#      filters = tables.Filters(complevel=5, complib='zlib') # 'zlib', 'bzip2', 'lzo'
#      hmm = h5.createCArray(h5.root, "test", tables.ComplexAtom(itemsize=16), self.conf[0]['Data']['X'].shape, filters=filters)
##         t = timeit.Timer(stmt=s)
#      hmm[:,:,:] = self.conf[0]['Data']['X']
#      h5.close()
         
