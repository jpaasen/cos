#from tables import openFile,IsDescription,createGroup,Int32Col,StrCol

#try:
#   from IPython import ColorANSI
#   from IPython.genutils import Term
#   tc = ColorANSI.TermColors()
#except:
#   pass

#from Operable import Operable

#import numpy as np
#import pylab as pl
import inspect

class dummy:
   def doh(self):
      pass

def is_builtin_scalar_type(typ):
   builtin_scalar_types = [float, long, int, complex]
   return typ in builtin_scalar_types

def is_builtin_type(typ):
   builtin_types = [float, long, int, complex, str, bool]
   return typ in builtin_types

def is_np_scalar_type(typ):
   import numpy as np
   #np_types = [
   #   np.int,   np.int8,  np.int16,  np.int32,   np.int64,
   #   np.uint,  np.uint8, np.uint16, np.uint32,  np.uint64,
   #   np.float,                      np.float32, np.float64,   np.float128,
   #   np.complex,                                np.complex64, np.complex128, np.complex256]
   
   np_types = [
      np.int,   np.int8,  np.int16,  np.int32,   np.int64,
      np.uint,  np.uint8, np.uint16, np.uint32,  np.uint64,
      np.float,                      np.float32, np.float64,
      np.complex,                                np.complex64, np.complex128]

   return typ in np_types

def is_np_array_type(typ):
   import numpy as np
   np_array_types = [np.ndarray]
   return typ in np_array_types or issubclass(typ, np.ndarray)

def is_np_type(typ):
   return is_np_scalar_type(typ) or is_np_array_type(typ)

def is_scalar(typ):
   return is_builtin_scalar_type(typ) or is_np_scalar_type(typ)

def is_array(typ):
   return is_np_array_type(typ)

#def is_np_or_builtin(typ):
#   return is_np_scalar_type(typ) or is_np_type(typ)
 


#fn = 0
#def newFig():
#   global fn
#
#   screen_res = [1920, 1080]
#
#   pl.figure()
#   figman = pl.get_current_fig_manager()
#   fig_geometry = figman.window.geometry().getCoords()
#   fig_width  = fig_geometry[2] #0.5*screen_res[0]
#   fig_height = fig_geometry[3] #0.7*screen_res[1]
#
#   max_x = int(pl.floor((screen_res[0]-fig_width)/20))
#   max_y = int(pl.floor((screen_res[1]-fig_height)/20))
#
#   disp_map = [(0, 60, fig_width, fig_height)]
#
#   for y in range(1,max_y):
#      for x in range(1,max_x):
#         if x != 1 or y != 1: #screen_res[1]-fig_height-
#            disp_map.append((20*x*y, 60+15*y*x, fig_width, fig_height))
#
#   dm = disp_map[fn]
#   fn += 1
#   print '%d %d %d %d' %(dm[0], dm[1], dm[2], dm[3]) 
#   figman.window.setGeometry(dm[0], dm[1], dm[2], dm[3]) # [0], 85, 800, 600
#   return


def processArgs(args, opts):
   """ Process optional parameters """
   Ni = len(args)
   if Ni != 0:
      for key,val in args:
         if opts.has_key(key):
            opts[key] = val
         else:
            print 'WW Operable.process_args() - Unknown argument '+key

   return opts

def error(calling_self, message):
   calling_method = 'unknown_caller()'
   stack = inspect.stack()
   print stack
   for i in range(0,stack.__len__()-1):
      if stack[i][3] == 'error':
         calling_method = stack[i+1][3]
         break
   try:
      print tc.BlinkRed+'EE: '+calling_self.__class__.__name__+'.'+calling_method+'(): '+message+tc.Normal
   except:
      print 'EE: '+calling_self.__class__.__name__+'.'+calling_method+'(): '+message
   
def warning(calling_self, message):
   calling_method = 'unknown_caller()'
   stack = inspect.stack()
   for i in range(0,stack.__len__()-1):
      if stack[i][3] == 'warning':
         calling_method = stack[i+1][3]
         break
   try:
      print tc.Yellow+'WW: '+calling_self.__class__.__name__+'.'+calling_method+'(): '+message+tc.Normal
   except:
      print 'WW: '+calling_self.__class__.__name__+'.'+calling_method+'(): '+message

def swarning(message):
   calling_method = 'unknown_caller()'
   stack = inspect.stack()
   for i in range(0,stack.__len__()-1):
      if stack[i][3] == 'warning':
         calling_method = stack[i+1][3]
         break
   try:
      print tc.Yellow+'WW: '+calling_method+'(): '+message+tc.Normal
   except:
      print 'WW: '+calling_method+'(): '+message

def info(calling_self, message):
   calling_method = 'unknown_caller()'
   stack = inspect.stack()
   for i in range(0,stack.__len__()-1):
      if stack[i][3] == 'info':
         calling_method = stack[i+1][3]
         break
   try:
      print tc.Green+'II: '+calling_self.__class__.__name__+'.'+calling_method+'(): '+message+tc.Normal
   except:
      print 'II: '+calling_self.__class__.__name__+'.'+calling_method+'(): '+message


#def save(system, desc=''):
#      
#   system.save(desc)


def save(self, desc=''):
   
   from Configurable import Configurable
   db = Configurable.__db__
                
   # Open the database
   db_root = db.open(type='system', mode='a')

   # Select the slot with the given description, or make it if it does not exist
   conf = db.open(type='configuration', mode='r')
   db.close('configuration')
   
   is_new = True
   if desc in conf:
      is_new = False


#      if desc in conf:
#         
#         # Remove the date and description
#         conf[desc].pop('date')
#         conf[desc].pop('desc')
#         
#         # Open the database
#         db_root = db.open(type='system', mode='a')
#         
#         for key in conf[desc]:
#            setattr(calling_module, key, conf[desc][key]['class_type']())
#            getattr(calling_module, key).load(db_root, conf[desc][key]['class_attr'])
#            
#         # Close the database
#         db.close()

   # Inspect the stack to find the syntax used to call save()
   stack = inspect.stack()
   for i in range(0,stack.__len__()-1):
      if stack[i][3] == 'save':
         calling_syntax = stack[i+1][4]
         break
   
   # Use regular expression to find the variable prior to save(), e.g. 's' in s.save()
   import re
   res = re.search('[a-zA-Z0-9\s]*(?=save)', calling_syntax[0])
   
   # Inspect the stack to find the syntax used to call save()
   calling_module = inspect.getmodule(stack[1][0])

   # Let the recursive fun begin..
   # Read the current configuration
   new_conf = {}
   for key,val in calling_module.__dict__.iteritems():
      try:
         if issubclass(val.__class__, Configurable):
            val.save(db_root, res.group(0), conf, new_conf, is_new)
      except:
         pass

   # Save and close the database
   db.flush()
   db.close()
   
   # Add the time of modification
   from datetime import datetime
   dt = datetime.today()
   new_conf['date'] = dt.strftime("%y.%m.%d - %H:%M:%S")
#      conf['id']   = int(time.mktime(dt.timetuple()))
   new_conf['desc'] = desc
   
   # Add the new contents
   conf[desc] = new_conf
     
   # Save the configuration back
   db.open(type='configuration', mode='w')
   db.dump(conf)
   db.close('configuration')


def load(desc=''):
      from Configurable import Configurable
      db = Configurable.__db__
      
#      import sys
#      main = sys.modules['__main__']
      
#      if cno == None:
#         error(self, 'You need to specify the configuration number to load.')
#         return
      
#      # Set default optional parameters
#      opts = {}
#      
#      # Process optional parameters
#      opts = processArgs(kwargs, opts)


      # Inspect the stack to find the syntax used to call save()
      stack = inspect.stack()
      calling_module = inspect.getmodule(stack[1][0])
      
#      for i in np.arange(0,stack.__len__()-1):
#         if stack[i][3] == 'load':
#            calling_syntax = stack[i+1][4]
#            break

      
      conf = db.open(type='configuration', mode='r')
      db.close('configuration')
      
     
      if desc in conf:
         
         # Remove the date and description
         conf[desc].pop('date')
         conf[desc].pop('desc')
         
         # Open the database
         db_root = db.open(type='system', mode='a')
         
         for key in conf[desc]:
            setattr(calling_module, key, conf[desc][key]['class_type']())
            getattr(calling_module, key).load(db_root, conf[desc][key]['class_attr'])
            
         # Close the database
         db.close()
          
      else:
#         error('', 'A configuration with the description \''+desc+'\' does not exist')
         return
            

def clear():
   import sys
   sys = sys.modules.clear()
   
#         self.__id__ = conf[desc]['s']['id']
#      else:

##      id = self.getConfId(desc, conf)
##      if id != None:
##         self.__id__ = id
##      else:
##         error(self, 'A configuration with the description \''+desc+'\' does not exist')
#               

#

#      
#      # Let the fun begin getattr(db_root,'i%d'%self.__id__)
#      Configurable.load(self, db_root, conf[desc])
#            

      

   

#def 
#   for key,val in self.__dict__.iteritems():
#      if issubclass(val.__class__, )

#def initLib(system):
#   """Creates and configures the library file."""
#   
#
#   
#   print h5f
#   
#   # Add another group for the data
#   data_grp = h5f.createGroup( '/', 'data', 'System data' )
#   
#
#   # Add more groups
#   for i in system.__dict__.iterkeys():
#      cname = i.__class__.__name__
#      data_obj_grp = h5f.createGroup( data_grp, cname, cname+' data' )
##      data_obj_tbl = h5f.createTable( data_obj_grp, 'Static data', DoubleTable )
#      data_obj_grp0 = h5f.createGroup( data_obj_grp, '0', 'Conf: 0' )
#      for key,val in i.__dict__.iteritems():
#         h5f.createCArray( data_obj_grp0, key, val )
#         
#   
#   
#   # Finally, flush the IO buffer to disk
#   conf_tbl.flush()
#   
#
#   
#   
##   gvar_grp = h5f.createGroup( '/', 'gvar', 'Global variables' )
