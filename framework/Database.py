#from Operable import Operable
#from tables import IsDescription, openFile, Filters, StringCol, Int32Col
import tables as tb
import numpy as np
from gfuncs import processArgs,error,info
#from Configurable import Configurable
import pylab as pl
import pickle
import os.path



class Database:
   """The global data collections. Whatever you need stored, make sure it's in this list.
   Regarding naming convention, large letters are used for matrices, small for vectors.
   
   Regarding compression: http://tukaani.org/lzma/benchmarks.html
     Best compression to decompression time ratio: zlib at level 5
     Best speed: lzo at level 9"""
   
   
   def __init__(self, *args):
      
      PHDCODE_ROOT = os.environ['PHDCODE_ROOT']
      # Process options
      opts = {'DATABASE_FILE':      PHDCODE_ROOT+'/data/database.hdf',
              'CONFIGURATION_FILE': PHDCODE_ROOT+'/data/configurations.pck'}
      opts = processArgs(args, opts)
      
      # Make a note of the DB location and whether it already existed
      self.DATABASE_FILE      = opts['DATABASE_FILE']
      self.CONFIGURATION_FILE = opts['CONFIGURATION_FILE']
      self.FILTERS = tb.Filters(complevel=9, complib='lzo') # 'zlib', 'bzip2', 'lzo'
      
      self.db_open = False
      self.conf_open = False
      
#      if os.path.exists(self.DATABASE_FILE):
#         self.db_is_new = False
#      else:
#         self.db_is_new = True
#         
#      if os.path.exists(self.CONFIGURATION_FILE):
#         self.conf_is_new = False
#      else:
##         self.conf_is_new = True
#
#         

   def updateConf(self, id, new_conf_entry):
         
      # Read the current configuration
      conf = self.open(type='configuration', mode='r')
      self.close()
            
      # Add the new contents
      conf[id] = new_conf_entry
        
      # Save the configuration back
      self.open(type='configuration', mode='w')
      self.dump(conf)
      self.close() 


   def open(self, type='system', mode='r'):

      
      if type == 'system':
         
         # Make sure the file is not open, and return the root node
         if not self.db_open:
            self.h5f = tb.openFile( self.DATABASE_FILE, mode=mode )
            self.db_open = True
            return self.h5f.getNode( self.h5f.root )
         else:
            error(self, 'System database is already open. Close it first.')
            
      elif type == 'configuration':
         
         # Create the file if it does not exist
         if not os.path.exists(self.CONFIGURATION_FILE):
            self.cf = open( self.CONFIGURATION_FILE, mode='w' )
            empty_dict = {}
            pickle.dump(empty_dict,self.cf)
            self.cf.close()
         
         # Make sure the file is not already open, then return the file pointer
         if not self.conf_open:
            self.cf = open( self.CONFIGURATION_FILE, mode=mode )
            self.conf_open = True
            if mode == 'r':
               try:
                  return pickle.load(self.cf)
               except:
                  info(self, 'Loaded empty file \''+self.CONFIGURATION_FILE+'\'')
                  return {}
         else:
            error(self, 'System database is already open. Close it first.')
                  
      else:
         error(self, 'Database type \''+type+'\' is unknown.')

#      if not self.h5f.isopen:
      
#      else:
#         print 'EE: '+self.__class__.__name__+'.open(): I\'m already open!'  
   
   def flush(self):
      if self.db_open == True:
         self.h5f.flush()
      else:
         error(self, 'Flushing not supported for this db-type.')
         
   def dump(self, data):
      if self.conf_open == True:
         pickle.dump(data, self.cf)
      else:
         error(self, 'This db-type does not support dumping.')
        
   
   def close(self, which='both'):
      
      if self.db_open and (which == 'both' or which == 'system'):
         self.h5f.close()
         self.db_open = False
         
      if self.conf_open and (which == 'both' or which == 'configuration'):
         self.cf.close()
         self.conf_open = False
            
      
#   def new(self, system):
#      
#      # Open the file and create the conf group
#      h5f = tb.openFile( self.DATABASE_FILE, mode='w' )
#      conf_grp = h5f.createGroup( '/', 'conf', 'System configurations' )
#      
#      # Create a column using a column description
#      conf_tbl = h5f.createTable(conf_grp, 'c0', ConfTable)
#      
#      # Append rows
#      newrow = conf_tbl.row
#      for val in system.__dict__.itervalues():
#         if issubclass(val.__class__, Configurable):
#            newrow['name'] = val.__class__.__name__
#            newrow['conf'] = val.conf
#            newrow['next'] = val.next
#            newrow.append()
#      
#      # Flush IO to file
#      h5f.flush()
#      h5f.close()

#   def getGroup(self, group, key):
#      try:
#         class_group = self.h5f.getNode(group, key)
#      except:
#         class_group = self.h5f.createGroup(class_group, key, title=obj_type, filters=self.FILTERS)

   def save(self, node, key, val, type='data'):
      '''
      obj_key      The variable that points to the object being stored
      obj_type     The type of the object being stored
      id      The configuration number being used
      keys         The variables in the object
      vals         and their values
      '''
               
      # If a groupname is supplied, open/create it and enter that group
      if type=='group' and pl.is_string_like(key):
         try:
            node = self.h5f.getNode(node, key)
         except:
            node = self.h5f.createGroup(node, key, filters=self.FILTERS)
      
      # If an id is supplied, make a group for it and enter that group
      elif type=='id' and pl.is_numlike(key):
         try:
            node = self.h5f.getNode(node, "i%d"%key)
         except:
            node = self.h5f.createGroup(node, "i%d"%key, filters=self.FILTERS)
      
      # When data is supplied, add it (and overwrite any data that was there before)
      elif type=='data':
         if key == None:
            key = 'None'
         elif not pl.is_string_like(key):
            error(self, 'This should not happen...')
                        
         try:
            self.h5f.removeNode(node, key)
         except:
            pass
         
         
         if issubclass(val.__class__, np.ndarray):
            
            # Empty tuples not supported by pytables. Fix by reshaping to (1,)
            if val.shape == ():
               val = val.reshape((1,))
                      
            atom = tb.Atom.from_dtype(val.dtype)
            new_node = self.h5f.createCArray(node, key, atom, val.shape, filters=self.FILTERS)
            new_node[:] = val[:]
            
         else:
            self.h5f.createArray(node, key, val)

         
      else:
         error(self, 'Hmm? Either \'type\' or \'key\' is of shitty format.')
         node = self.h5f.root
 
      
         
      return node

#   def last_modified(self):
#      return self.last_modified
   
#         if type(val) == int:
#
#         elif type(val) == float:
#            try:
#               entry = self.h5f.getNode(conf_group, key)
#            except:
#               entry = self.h5f.createArray(conf_group,key,val,filters=self.FILTERS)
#         elif type(val) == ndarray:
#            try:
#               entry = self.h5f.getNode(conf_group, key)
#            except:
#               entry = self.h5f.createArray(conf_group,key,val,filters=self.FILTERS)
#         entry[:] = val
      

#      h5.createCArray(h5.root, "test", tables.ComplexAtom(itemsize=16), self.conf[0]['Data']['X'].shape, filters=filters)
##         t = timeit.Timer(stmt=s)

#      class_desc = []
#      for var,val in i.__dict__.itervalues():
#         if type(var) == None:
#            pass
#         else:
#            pass
               
#               h5.createTable( group,   )
#      # Open the file
#      h5 = tables.openFile(CONF_FILE, mode = "w")
#      hmm = h5.createCArray(h5.root, "test", tables.ComplexAtom(itemsize=16), self.conf[0]['Data']['X'].shape, filters=filters)
##         t = timeit.Timer(stmt=s)
#      hmm[:,:,:] = self.conf[0]['Data']['X']
#      h5.close() 

      
