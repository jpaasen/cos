# coding=<encoding name>


from Configurable import Configurable

from gfuncs import error

from datetime import datetime
     

class System(Configurable):
   """A system is a collection of class instances that share the same configuration."""
   
   
   def __init__(self, desc=''):

#      self.window    = Window()
#      self.delay     = Delay()
#      self.signal    = Signal() 
#      self.interface = Interface()
#      self.array     = Array()
#      self.element   = Element()
#      self.medium    = Medium()
#      self.coords    = Coordinate()
      
#      self.data      = Data()     
      
#      self.calc      = Calc(self)
#      self.plot      = Plot(self)
      
      if desc != '':
         self.load(desc)
      
   
   def confexists(self, desc):
      db = Configurable.__db__
      
      conf = db.open(type='configuration', mode='r')
      db.close('configuration')
      
      if not desc in conf:
         return False
      
      return True

   
   def __getitem__(self, *keys):
      no_keys = keys.__len__()
      if no_keys == 0:
         return
      elif no_keys == 1:
         return Configurable.findAttr(self, keys[0])
      else:
         vals = []
         for key in keys:
            vals.append(Configurable.findAttr(self, key))
         return vals
      
   
   def save(self, desc=''):

      if desc == '':
         if 'desc' in self.__dict__:
            desc = self.desc
      else:
         self.desc = desc

      
      # Save data to DB
      # Save configuration to Conf
      db = Configurable.__db__
             
      # Select the slot with the given description, or make it if it does not exist
      full_conf = db.open(type='configuration', mode='r')
      db.close('configuration')
      
      if desc in full_conf:
         conf = full_conf[desc]
      else:
         conf = None

      # Let the recursive fun begin..
      # Read the current configuration
      db_root = db.open(type='system', mode='a')
      new_conf = {}
      Configurable.__save__(self, db_root, 's', conf, new_conf)
      
      # Save and close the database
      db.flush()
      db.close()
      
      # Now, let us save the corresponding configuration
#      new_conf = new_conf.pop('%d'%self.__id__)
#      new_conf = new_conf.pop('class_attr')
      
      # Add the time of modification
      dt = datetime.today()
      new_conf['date'] = dt.strftime("%y.%m.%d - %H:%M:%S")
#      conf['id']   = int(time.mktime(dt.timetuple()))
      new_conf['desc'] = desc
            
      # Add the new contents
      full_conf[desc] = new_conf
        
      # Save the configuration back
      db.open(type='configuration', mode='w')
      db.dump(full_conf)
      db.close('configuration')


   def load(self, desc=''):
      db = Configurable.__db__
      
      conf = db.open(type='configuration', mode='r')
      db.close('configuration')
      
      if not desc in conf:
         error(self, 'A configuration with the description \''+desc+'\' does not exist')
         raise Exception('')
         return

#      id = self.getConfId(desc, conf)
#      if id != None:
#         self.__id__ = id
#      else:
#         error(self, 'A configuration with the description \''+desc+'\' does not exist')
               
      # Remove the date and description
      conf[desc].pop('date')
      conf[desc].pop('desc')

      # Open the database
      db_root = db.open(type='system', mode='a')
      
      # Let the fun begin getattr(db_root,'i%d'%self.__id__)
      Configurable.__load__(self, db_root, conf[desc]['s'])
            
      # Close the database
      db.close() 

      Configurable.__refreshEclipse__(self)
      # Inspect the stack to find the syntax used to call save()
#      stack = inspect.stack()
#      for i in np.arange(0,stack.__len__()-1):
#         if stack[i][3] == 'load':
#            calling_syntax = stack[i+1][4]
#            break
#         
#      # Use regular expression to find the variable prior to save(), e.g. 's' in s.save()
#      res = re.search('[a-zA-Z0-9]*(?=\.load)', calling_syntax[0])
