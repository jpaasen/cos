#def num2str
from genericpath import exists

global FLAGS
from copy import deepcopy,copy
from numpy import sqrt,vstack
from scipy import io

class FileRegister:
   desc   = ''
   type   = ''
   group = ''
   index = 0
   date   = ''
   file   = ''
   path   = ''
   # lpath, rhost, and rpath are just temporaries, thus are not defined statically
   
   def __init__(self, data_register = None ):
      if data_register is None:
         pass
      else:
         self.desc    = data_register.desc
         self.type    = data_register.type
         self.group   = data_register.group
         self.index   = data_register.index
         self.date    = data_register.date 
         self.file    = data_register.file 
         self.lpath   = data_register.lpath
         self.rhost   = data_register.rhost
         self.rpath   = data_register.rpath
         if hasattr(data_register, 'info_file'):
            self.info_file = data_register.info_file

def addUsMultiFileDataset(dataContainer, nFrames, desc, group, date, filePrefix, path, rhost):
   s = FileRegister()
   s.desc  = desc
   s.type  = 'ultrasound multi file'
   s.group = group
   s.index =  0
   s.date  = date
   s.file  = filePrefix
   s.lpath = '/data/' + path
   s.rhost = rhost
   s.rpath = '/not/available/'
   
   si = FileRegister(s)
   si.desc += ' info'
   si.file += 'info.mat'
   dataContainer.append(FileRegister(si))
   
   info_file_name = si.file
   
   for i in range(1,nFrames+1):
      si = FileRegister(s)
      si.info_file = info_file_name
      si.index = i
      si.desc   += ' ' + "%d"%i
      si.file   += "%dframe.mat"%i
      dataContainer.append(FileRegister(si))
   

def fileLUT( group, index, *args ):
   """
   This function has one aim: To list all relevant datasets that may be used
   with the Dataset class. For each dataset the following parameters should
   be supplied:
   
   desc   : A short description of the dataset
   class : The class (e.g. 'hugin') indirectly specifies the dataset format
   group : Several datasets may naturally belong to some group...
   index : ..these are identified with incementing indexes
   date   : When was the data acquired?
   file   : The filename
   lpath : Local path, i.e. the path on your computer
   rhost : Remote host where the dataset can be found,
   rpath : and the path to it.
   """
   
   # TODO: As is, this list is generated each time the LUT is called. Improvements are
   # definitely possible.

   #Just list all valid datapaths
   data            = []
   
   ## Below here is ultrasound datasets
   
   # simulation of motion phantom 16x oversampling   
   addUsMultiFileDataset(data, 50, 'Field II simulation', 'motion phantom 16x 3.4 MHz 20dB30dB', '2013', 
                         'motion_phantom_16x_', 
                         'motion_phantom_16x_3-4MHz_20dB30dB/', 
                         'Field II')
   
   result = 0
   for i in data:
      bREMOTE_PATH = False
      if i.group == group and i.index == index:
         for j in args:
            if j == 'remote':
               bREMOTE_PATH = True
               
         if bREMOTE_PATH:
            i.path       = i.rpath
            bREMOTE_PATH   = False
         else:
            i.path       = i.lpath

         del i.lpath
         del i.rpath
         del i.rhost
         
         result = deepcopy(i)
         break
      
   if result != 0:
      return result
   else:
      print 'WW dataLUT(): ''group'' or ''index'' was fishy. Omitting.' 
   
