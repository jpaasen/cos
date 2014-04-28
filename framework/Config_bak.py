# -*- coding: utf-8 -*-


'''
Created on Feb 10, 2011

@author: me
'''

from fileLUT import fileLUT
#from numpy import *
from scipy import io
from Configurable import Configurable
import sys
import pickle
from mynumpy import Ndarray
import mynumpy as np
#import mynumpy as np 
from copy import copy

#import cPickle as pickle
#import sqlite3
#import shelve
#from tables import * # You'll need pytables+hdr5 installed for this to work 
global gvars

class Xd(Configurable):
   def __init__(self, Xd):
      self.Xd = Xd
   def getData(self):
      return self.Xd
   
class p(Configurable):
   pass

#try:
#   dyn_cls = pickle.load(open('data/Config_dynamic_types.pck','r'))
#   for key in dyn_cls:
#      setattr(sys.modules['Config'], key, dyn_cls[key])
#except:
#   pass
#
#dyn_cls = {}
## Create a 'cameleon class' that can take on any type
#def cls(key, val=None, desc=None, shape_desc=None):
#   new_class = type(key,(Operable,Configurable),{})
#   setattr(sys.modules['Config'], key, new_class)
#   dyn_cls[key] = new_class
#   pickle.dump(dyn_cls,open('data/Config_dynamic_types.pck','w'))
#   return getattr(sys.modules['Config'], key)(key,val,desc,shape_desc)
#
## Dynamic class
#class DC(Operable, Configurable):
#   
#class D(Configurable, Operable):
#   def __init__(self, key, val=None):
#      self.__class__.__name__ = key
#      setattr(sys.modules['Config'], key, type(key,(object,),{}))
#      Operable.__init__(self,key,val)

      
#class DataWrapper(Operable, Configurable):

   
#
#class ConfTable(IsDescription):
#   name = StringCol(itemsize=16, pos=0)
#   conf = Int32Col(pos=1)
#   next = StringCol(itemsize=16, shape=2, pos=2, dflt=('-','-'))

class Config():
   """
   The purpose of this class is to create a single dataset structure
   regardless of the input data being used. In particular, the following
   specifications are upheld:
   
   Coordination system
   Regardless of the input data, the output coordination system will have axes
   coinciding with the array layout, and origin in the array phase center. In
   Ultrasound this is the case per default, but e.g. HUGIN body-coordinates are
   different from the coordination system of the mounted HISAS1030 array.
      
      Stores the following 
      _desc                         # Dataset description
      _origin                      # Info from dataLUT() regarding data origin
      _array                        # ApertureArray object

      _s                              # Sensor/detector struct
#    s.c                           # Propagation speed
#    s.fc                         # Carrier frequency      

      _d                              # Data struct
#    d.X                           # Raw data [ N_samples, M_sensors ]
#    d.t                           # Corresponding time axis [N_samples]
#    d.Xinfo                     # Data info ('raw', 'match filtered', ...)
#    d.Xd                         # Delayed data [ N_samples, M_sensors ]
   """
   # The configuration is just a flat dictionary of all system classes, which each of
   # these entries being another dictionary with the configuration for that particular
   # class.

   
   def __init__(self):
      self.conf = {}
      self.id   = None
      self.Nid  = 0

   def __getitem__(self, key):
      return self.conf[key]
   
   def has_key(self, key):
      return self.conf.has_key(key)
   
   def loadFile(self, group, index, *args):
      '''Load configuration from a predefined list of .mat files'''
      
      from Coordinate import Coordinate
      from Window import Window
      from Medium import Medium
      from Signal import Signal
      from System import System
      from Data import Data
      from Array import Array

      from Image import Image
      
      import os
      PHDCODE_ROOT = os.environ['COS_ROOT']
      
      Ni = len(args)
   
      if type(index) == int:
         idx = copy(index)
         index = [idx]
         
      for i in index:
         if Ni > 0:
            origin = fileLUT(group, i, args)
         else:
            origin = fileLUT(group, i)
      
    
      if origin.type == 'ultrasound multi file':  
        
         print 'Loading ' + PHDCODE_ROOT + origin.path + origin.file
         #print 'Loading ' + PHDCODE_ROOT + origin.path + origin.info_file
         info_file = io.loadmat(PHDCODE_ROOT + origin.path + origin.info_file)
         mat_file = io.loadmat(PHDCODE_ROOT + origin.path + origin.file)
         
         NFrames   = info_file['NFrames'][0,0]
         NElements = info_file['NElements'][0,0]
         angles = info_file['angles']
         ranges = info_file['ranges']
         
         tmp = mat_file['frame%d'%origin.index].shape
         
         Xd = np.zeros((tmp[0], tmp[1], tmp[2]), dtype=np.complex64)
         Xd[:,:,:] = mat_file['frame%d'%i]
         
         s = System()

         s.data = Data()
         s.data.Xd      = Xd
         s.data.angles  = angles
         s.data.ranges  = ranges
         s.data.M = NElements
         s.data.NFrames = NFrames

         return s
         
      elif origin.type == 'ultrasound_simulation':
         
         import tables as tb
         s = System()
         s.data = Data()

         print 'Loading ' + PHDCODE_ROOT + origin.path + origin.file
         
         file = tb.openFile( (PHDCODE_ROOT + origin.path + origin.file), 'r' )
             
         root = file.getNode( file.root )

         s.data.M       = root.frame_1.N_rx.read()
         s.data.angles  = root.frame_1.theta.read()[0]
         s.data.ranges  = root.frame_1.range.read()
                  
         d = root.frame_1.datacube.read().shape
         s.data.Xd = np.zeros((100,d[1],d[2],d[0]),dtype=np.complex64)
         
         for i in range(100):
            tmp = np.transpose(getattr(root, 'frame_%d'%(i+1)).datacube.read(), (1, 2, 0))
            s.data.Xd[i,:,:,:] = tmp
            
         file.close()
         
         return s
         
      else:
         print 'WW DataInput() - No method implemented for '+origin.type

   
      
   def loadTemplate(self, key):
      
      conf = {}
      
      if key == '1D ULA, water, critical sampling':
         
         # List the parameters we want:
         M     = 32        # Number of elements
         c     = 1540      # Acustic propagation speed in water
         fc    = 10e3      # Carrier frequency
         lmda  = c/fc      # Wavelength
         d     = lmda/2    # Distance between elements (spatial sampling)
#         w     = ones(M)/M # A rectangular window
         
         # Create some axes
         p     = linspace(0,1,M)*M*d
         
         # Find
#         kx  = k*cos(theta)

         conf['Medium'] = {
#            'id'       : 0                 ,
            'type'          : 'simple'          ,
            'c'             : c                 }

         conf['Interface'] = {
#            'id'       : 0                 ,
            'type'          : 'simple'          }


         p = Coordinate(x=p, y=None, z=None) 

         conf['Array'] = {
#            'id'       : 0                 ,
            'type'          : 'simple'          ,
            'p'             : p                 }

         conf['Data'] = {
#            'id'       : 0                 ,
            'type'          : 'none'            }

         w = Window()
         w.kaiser(32, 1, 0)

         conf['Window'] = {
#            'id'       : 0                 ,
            'type'          : 'custom'          ,
            'w'             : w.getData()       }
         
         conf['Signal' ] = {
#            'id'       : 0                 ,
            'type'          : 'continuous wave' ,
            'fc'            : fc                }

         conf['System' ] = {
#            'id'       : 0                 ,
            'type'          : 'Simple HUGIN template',
            'next'          : ('Window', 'Window')   }
      
      self.conf = conf

