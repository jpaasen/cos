from Configurable import Configurable
from mynumpy import linspace 

class Array(Configurable):
   def __init__(self, *args):
#      self.type      = None       # Array type
#      self.prop      = None       # Properties (depends on array type)
#      self.p          = None       # Element coordinates (rank of matrix specifies number of dimensions)
#      Operable.__init__(self, 'p')
      pass
                  
   def configure(self, conf):       
      if conf.has_key('p'):
         self.type       = 'custom'
         self.p            = conf['p']
      elif conf.has_key('M') and conf.has_key('d'):
         self.type       = 'ula'
         self.type       = type                                           # Store type
         self.prop.M    = conf['M']                                    # Number of elements
         self.prop.d    = conf['d']                                    # Element distance
         D                   = conf['M']*conf['d']                     # Array length
         self.p            = linspace( -D/2, D/2, conf['M'] ) # Element positions
      else:
         print 'WW Array.configure() - Neither \'p\', \'M\', nor \'d\' was specified.'
         
#   def __getitem__(self, key):
#      return getattr(self, key)
         
