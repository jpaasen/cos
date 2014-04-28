from gfuncs import error, warning
import pylab as pl
from Database import Database


# We let the Configurable class inherit from 'object' such that all the other
# classes that inherit from Configurable becomes new style classes. Google it. :)
class Configurable(object):
   
   __db__ = Database()
   
 
   """
   This class provides the functionality that allows classes in the system
   to __save__ and __load__ themselves.
   """
   
   # loadConf() recursively loads configuration to all lower level classes
   def loadConf(self, conf, *args):
      
      # Check the supplied object type
      if conf.__class__.__name__ != 'Config':
         error(self, 'Supplied object is not of type \'Config\'')
         return
      
      # Process the optional arguments
#      default_opts = {}
#      opts = processArgs(args, default_opts)
      
      ok_list = ''
      fail_list = ''
      has_id = False
      
      # Iterate over all class instance variables
      for key,val in self.__dict__.iteritems():
         
         # If the variable is another configurable class, call its loadConf()
         if issubclass(val.__class__, Configurable):
            val.loadConf(conf)
         else:
            
            # Check that a configuration record exists for this class
            self_name = self.__class__.__name__
            if not conf.has_key(self_name):
               warning(self, 'This class has no conf-record.')
            
            # If it exists, retrieve the data
            else:
               if conf[self_name].has_key(key):
                  setattr(self, key, conf[self_name][key])
                  ok_list += ' '+key
                  if key == 'id':
                     has_id = True
               else:
                  fail_list += ' '+key
      
      warning(self,'')
      print '    OK: '+ok_list
      print '    NA: '+fail_list#   '+self_name+': '+key
      
       
         
   def findAttr(self, attr):
   
      if attr in self.__dict__:
         return getattr(self, attr)
      
      else:      
         # Iterate over all class instance variables
         for key,val in self.__dict__.iteritems():
            
            # If the variable is another configurable class, call its getConfNo()
            if issubclass(val.__class__, Configurable):
   
               res = val.findAttr(attr)
               if res != None:
                  return res

      return None
   
   
   def __refreshEclipse__(self):
      
                  
      # Iterate over all class instance variables
      for key,val in self.__dict__.iteritems():
         
         # If the variable is another configurable class, call its getConfNo()
         if issubclass(val.__class__, Configurable):
            val.__refreshEclipse__()
            
         else:
            delattr(self, key)
            setattr(self, key, val)

   
   def __find_new_slot__(self, root_node, conf):
      import mynumpy as np
      
      dummy_conf = {}
      db = Configurable.__db__
      
      class_node = db.save(root_node, self.__class__.__name__, None, type='group')
      
      # If the configuration is new, set a new slot as default
      if conf == None:
         id = class_node._v_nchildren
         
      # If a configuration exists, set the old slot as default
      else:
         id = conf['id']
         conf = conf['class_attr']
         
      # Iterate over each of the index slots in the class
      is_equal = True
      for i in range(0,class_node._v_nchildren):
         
         id_node = db.save(class_node, i, None, type='id')
         is_equal = True
         
         # Iterate over all class variables
         for key,val in self.__dict__.iteritems():
            
            if val != None:
               cn = val.__class__.__name__
               vc = val.__class__
   
               # For configurable types, just make sure their entry is the DB
               if issubclass(vc, Configurable):
                  if cn not in root_node:
                     is_equal = False
                     break
                  
               # The values of common types need to be compared aswell
               if vc == float or vc == long or vc == int or vc == complex or vc == str:
               
                  # Fetch the data if it exists
                  if key in id_node:
                     db_val = getattr(id_node, key)
                  else:
                     is_equal = False
                     break
                  
                  
                  # Compare the values. If they are unequal, or the compare operation
                  # is undefined for the two types, the values are treated as unequal.
                  try:
                     if val != db_val.read():
                        is_equal = False
                        break
                  except:
                     is_equal = False
                     break
                  
               # Reading and comparing all arrays in the database is performance
               # intensive. To speed up the process the DB entry is only read if the
               # matrix dimensions are equal:
               elif issubclass(vc, np.Ndarray):
                  
                  # Fetch the data if it exists
                  if key in id_node:
                     db_val = getattr(id_node, key)
                  else:
                     is_equal = False
                     break
                  
                  # Record the array lengths
                  val_len = val.shape.__len__()
                  db_val_len = db_val.shape.__len__()
                  
                  # Are the values scalars?
                  if val_len==0 or (val_len==1 and val.shape==(1,)):
                     if db_val_len==0 or (db_val_len==1 and db_val.shape==(1,)):
                        db_val = db_val.read()
                        if val != db_val:
                           is_equal = False
                           break
   
                  # ...or arrays?
                  else:
                     # Make sure the DB entry is an array-type 
                     if db_val_len>1 or db_val.shape[0]>1:

                        if db_val.shape == val.shape:
                           db_val = db_val.read()
                           t = val != db_val
                           if (np.isscalar(t) and t) or t.any():
                              is_equal = False
                              break
                        else:
                           is_equal = False
                           break
                        
               elif vc == tuple or vc == list:
                  
                  val_len = val.__len__()
                  db_val = []
                  
                  # Create the wrapper object and fill it with list/tuple items
                  if vc == tuple:
                     tw = TupleWrapper()
                  else:
                     tw = ListWrapper()
                  
                  for j in np.arange(0,val_len):
                     setattr(tw, 'i%d'%j, val[j])
                     
                  # Call its find new slot to recursively dig into the data
                  was_equal,id2 = tw.__find_new_slot__(root_node, conf)
                  
                  # If any of the list items were unequal, we return
                  if was_equal == False:
                     is_equal = False
                     break
                           
   
         # If all contents are equal to that stored at a specific slot in the DB,
         # set the id of this class to match that DB slot. 
         # TODO: Performance tip: I'm storing thing twice now.. 
         if is_equal:
            id = i
            break
         
      return is_equal,id
   
         
   def __save__(self, root_node, self_key, conf, new_conf):
      '''
      self_key    The variable in the calling class that points to self
      db          The database
      depth       The recursive level'''
      import mynumpy as np
      from gfuncs import is_builtin_type, is_np_type
      # Process the optional arguments
#      opts = {'is_subclass':False}
#      opts = processArgs(args, opts)
      
      db = Configurable.__db__
      
      # Start by adding the key for this class, if supplied.
      if self_key != None and pl.is_string_like(self_key):
         new_conf[self_key] = {}
         new_conf = new_conf[self_key]
         new_conf['class_type'] = self.__class__
         class_node = db.save(root_node, self.__class__.__name__, None, type='group')
         
         if conf != None:
            conf = conf[self_key]
              
      # Then add the configuration number
      # Check if the data already exists (in which case a new slot is not needed)
      
      # If this is the first configuration, we'll __save__ the data at the first slot
     
      
      # If one or more configurations exist, and the 'is_new' flag is given, we
      # select a slot with equal contents, if such a slot exists. If not, a new
      # slot is created and the data is saved there.
      
      # To ensure that no data is stored twice, this search must be performed
      # each time.

      was_equal, id = self.__find_new_slot__(root_node, conf)
      new_conf['id'] = id
      new_conf['class_attr'] = {}
      new_conf = new_conf['class_attr']
      id_node = db.save(class_node, id, None, type='id')
      
     
      # Iterate over all class instance variables
      for key,val in self.__dict__.iteritems():
         vc = val.__class__
         
#         print key
         
         #if key=='Xd':
         #   print 'hello'
        
         # Skip empty types
         if val == None:
            pass

         # If the variable is another configurable class, call its __save__()
         if issubclass(val.__class__, Configurable):
            if vc == np.Ndarray:
               db.save(id_node, key, val, type='data')
               if key not in new_conf:
                  new_conf[key] = val.__class__
            else:
               val.__save__(root_node, key, conf, new_conf)
         
         # Save normal datatypes to the database
         if is_builtin_type(vc) or is_np_type(vc):
            
            if key == None:
               key = 'None'
            
            db.save(id_node, key, val, type='data')
            if key not in new_conf:
               new_conf[key] = val.__class__
            
         # NOTE: Be careful with this one. Prolly try to avoid these types
         elif type(val) == tuple or type(val) == list:
            
            if type(val) == tuple:
               tw = TupleWrapper()
            else:
               tw = ListWrapper()
               
            tw.len = val.__len__()
            
            for i in range(0,tw.len):
               if val[i] == None:
                  val[i] = 'None'
               setattr(tw, 'i%d'%i, val[i])
            
            tw.__save__(root_node, key, conf, new_conf)
            
         elif type(val) == dict:
            
            dw = DictWrapper()
            
            for subkey in val:
               if subkey == None:
                  setattr(dw, 'None', val[subkey])
               else:
                  setattr(dw, subkey, val[subkey])
            
            dw.__save__(root_node, key, conf, new_conf)
            

   def __load__(self, root_node, conf):
      import mynumpy as np

      import Coordinate
      '''
      self_key    The variable in the calling class that points to self
      db          The database
      depth       The recursive level'''
#      from TypeExtensions import Ndarray
      
      class_node = getattr(root_node, self.__class__.__name__)
      id = conf.pop('id')
      node = getattr(class_node,'i%d'%id)
      conf = conf['class_attr']
      

      # Iterate over all class instance variables
      for key in conf:
         
         # Lookup the type of the variable
         try:
            class_type = conf[key].pop('class_type')
         except:
            class_type = conf[key]
         
            
         # Configurable types are first created, then asked to load themselves
         if issubclass(class_type, Configurable):
            
            # Hmm, this is some nasty piece of code; A 'lookahead' to fetch the
            # data of the base class in order to initialize the Ndarray with
            # that data.. 
            if class_type == np.Ndarray:
               data = getattr(node,key).read()
#               tmp = class_type(data)
#               tmp.__load__(root_node, conf[key])
#               axes = tmp.__dict__.pop('axes')
#               setattr(self, key, class_type(data, axes))
               setattr(self, key, class_type(data))
#               for subkey in tmp.__dict__:
#                  setattr(getattr(self, key), subkey, getattr(tmp, subkey))

            # TODO: Should make this more general, but no time for that now
            elif issubclass(class_type, Coordinate.Coordinate):
               data = getattr(node,key).read()
               setattr(self, key, class_type(x=data[:,0],y=data[:,1],z=data[:,2]))

            else:
               setattr(self, key, class_type())
               getattr(self,key).__load__(root_node, conf[key])
            
               
            # Unwrap tuples and lists if necessary:            
            if class_type == TupleWrapper or class_type == ListWrapper:
               val = []
               for i in np.arange(0,getattr(self,key).len):
                  subval = getattr(getattr(self,key), 'i%d'%i)
                  if subval == 'None':
                     subval = None
                  val.append(subval)
               delattr(self,key)
               if class_type == TupleWrapper:
                  val = tuple(val)
               setattr(self, key, tuple(val))
               
            elif class_type == DictWrapper:
               val = {}
               for subkey in self.__dict__:
                  if subkey == 'None':
                     val[None] = getattr(getattr(self,key),'None')
                  else:
                     val[subkey] = getattr(getattr(self,key), subkey)

               delattr(self,key)
               setattr(self, key, val)
               

         else:           
            setattr(self, key, getattr(node,key).read())
            
            
class TupleWrapper(Configurable):
   pass
class ListWrapper(Configurable):
   pass
class DictWrapper(Configurable):
   pass

