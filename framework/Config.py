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
      
      # The dataset 'type' (defined by me) selects the "import method":
      if origin.type == 'hugin_delayed':
         
         # Using scipy.io.loadmat() to load .mat files. The result is a nparray, where structures
         # in the array are dictionaries (the data is treated later on..). 
         print 'Loading ' + PHDCODE_ROOT + origin.path + origin.file
         mat_file = io.loadmat(PHDCODE_ROOT + origin.path + origin.file)

         BF = mat_file['p'][0, 0]['BF'][0, 0]
         
         s = System()
         
         s.data = Data()         
         s.data.Xd = Ndarray(mat_file['dataCube'].transpose((1,0,2)))

         s.data.n = Ndarray(BF['mtaxe_re' ].T)

         s.data.Nt = Ndarray(BF['Nt'].T)
         s.data.fs = 40e3
         s.data.fc = 100e3
         s.data.M  = 32
         s.data.d  = 0.0375
         s.data.c  = 1500
                  
         s.desc = 'HUGIN raw data'
         
         return s
         
      elif origin.type == 'focus_dump_type_A' or origin.type == 'focus_dump_type_B':
            
         # Using scipy.io.loadmat() to load .mat files. The result is a nparray, where structures
         # in the array are dictionaries (the data is treated later on..). 
         print 'Loading ' + PHDCODE_ROOT + origin.path + origin.file
         mat_file = io.loadmat(PHDCODE_ROOT + origin.path + origin.file)
         
         # Load the 'Sonar' dictionary and get rid of empty dimensions:
         Sonar = mat_file['Sonar'][0,0]
         
         s = System()
         
         s.medium = Medium()

         s.medium.c = Ndarray(Sonar['c'][0,0])    # Propagation speed
      
       
#            self.B                = 0                                 # Assume narrowband
         
         # The 'rx_x','rx_y' and 'rx_z' coordinates in the Sonar struct represents
         # the TX elements' location relative to the HUGIN body. We're going to
         # rotate this coordination system around the x-axis such that the
         # y-coordinate is 0 for all elements (22 degrees). 
         
         rx_x = Sonar['rx_x'][:,1]           # RX x-coordinates (HUGIN body)
         rx_y = Sonar['rx_y'][:,1]           # RX y-coordinates (HUGIN body)
         rx_z = Sonar['rx_z'][:,1]           # RX z-coordinates (HUGIN body)
         rx_yz = np.sqrt(rx_y ** 2 + rx_z ** 2)    # RX y-z distance (new z-axis)
         p = Coordinate(x=rx_x, y=None, z=rx_yz)# New RX coordinates [x, axis]
         p.setAxesDesc(template='hugin')
         s.array = Array()
         s.array.p = p
#                             desc='RX coordinates',
#                             shape_desc=('x','axis'))
                 
         s.signal = Signal()
         s.signal.desc = 'upchirp'
         s.signal.fc   = Ndarray(Sonar['fc'][0,0])
         
         
         if origin.type == 'focus_dump_type_A':

            s.data = Data()
            s.data.X = Ndarray(mat_file['mfdata'].T.copy())
            s.data.t = Ndarray(mat_file['mtaxe' ][0])
            
            s.image = Image()
            s.image.xarr = Ndarray(mat_file['xarr'].T.copy())
            s.image.yarr = Ndarray(mat_file['yarr'].T.copy())
            
            
         else:
            
            s.data = Data()
            s.data.X = Ndarray(mat_file['data1'].T.copy())
            s.data.t = Ndarray(Sonar['T_pri'][0,0])

            s.image = Image()
#            s.image.xarr = Ndarray(mat_file['xarr'].T.copy())
#            s.image.yarr = Ndarray(mat_file['yarr'].T.copy())


         return s


      elif origin.type == 'focus_sas_dump_type_A_parameters':
            
         # Using scipy.io.loadmat() to load .mat files. The result is a nparray, where structures
         # in the array are dictionaries (the data is treated later on..). 
         print 'Loading ' + PHDCODE_ROOT + origin.path + origin.file
         mat_file = io.loadmat(PHDCODE_ROOT + origin.path + origin.file)
         
         # Load the 'Sonar' dictionary and get rid of empty dimensions:
         Sonar = mat_file['Sonar'][0,0]
         
         s = System()
         
         s.medium = Medium()
         s.medium.c = Ndarray(Sonar['c'][0,0])    # Propagation speed
      
       
#            self.B                = 0                                 # Assume narrowband
         
         # The 'rx_x','rx_y' and 'rx_z' coordinates in the Sonar struct represents
         # the TX elements' location relative to the HUGIN body. We're going to
         # rotate this coordination system around the x-axis such that the
         # y-coordinate is 0 for all elements (22 degrees). 
         
         rx_x = Sonar['rx_x'][:,1]           # RX x-coordinates (HUGIN body)
         rx_y = Sonar['rx_y'][:,1]           # RX y-coordinates (HUGIN body)
         rx_z = Sonar['rx_z'][:,1]           # RX z-coordinates (HUGIN body)
         rx_yz = np.sqrt(rx_y ** 2 + rx_z ** 2)    # RX y-z distance (new z-axis)
         p = Coordinate(x=rx_x, y=None, z=rx_yz)# New RX coordinates [x, axis]
         p.setAxesDesc(template='hugin')
         s.array = Array()
         s.array.p = p
#                             desc='RX coordinates',
#                             shape_desc=('x','axis'))
                 
         s.signal = Signal()
         s.signal.desc = 'upchirp'
         s.signal.fc   = Ndarray(Sonar['fc'][0,0])
         
         return s
      

      elif origin.type == 'focus_sas_dump_type_A_data':
            
         # Using scipy.io.loadmat() to load .mat files. The result is a nparray, where structures
         # in the array are dictionaries (the data is treated later on..). 
         print 'Loading ' + PHDCODE_ROOT + origin.path + origin.file
         mat_file = io.loadmat(PHDCODE_ROOT + origin.path + origin.file)
         
        
         s = System()
         
         s.data = Data()
         s.data.X = Ndarray(mat_file['mfdata'].copy())
         s.data.t = Ndarray(mat_file['mtaxe' ][0])
            
         return s

      elif origin.type == 'focus_sss_dump_type_A':
            
         # Using scipy.io.loadmat() to load .mat files. The result is a nparray, where structures
         # in the array are dictionaries (the data is treated later on..). 
         print 'Loading ' + PHDCODE_ROOT + origin.path + origin.file
         mat_file = io.loadmat(PHDCODE_ROOT + origin.path + origin.file)
         
        
         s = System()
         
#         s.a = Data()
         s.img = Ndarray(mat_file['sss_image'].copy())
            
         s.Nping = 140
         s.Ny    = 4000
         s.Nx    = 15
         
         s.y_lim = [60,160]
         
            
         return s

            
      elif origin.type == 'hisas_sim':

         # Using scipy.io.loadmat() to load .mat files. The result is a nparray, where structures
         # in the array are dictionaries (the data is treated later on..). 
         print 'Loading ' + PHDCODE_ROOT + origin.path + origin.file
         mat_file = io.loadmat(PHDCODE_ROOT + origin.path + origin.file)

         BF = mat_file['p'][0, 0]['BF'][0, 0]
         
         s = System()
         
         s.medium = Medium()
         s.medium.c = Ndarray(BF['c'][0, 0])     # Propagation speed
         
         s.array = Array()
         s.array.desc = 'ula'
         s.array.M    = Ndarray(BF['n_hydros'][0, 0])  # Number of hydrophones
         s.array.d    = Ndarray(BF['d'       ][0, 0])     # Element distance

         s.data = Data()
         s.data.Xd = Ndarray(mat_file['dataCube']).transpose((1,0,2))
#         Ndarray(,
#                             axes = ['y','x','m'],
#                             desc = 'Delayed and matched filtered data')
#                            desc='Delayed and matched filtered data',
#                            shape_desc = ('y','x','m'),
#                            shape_desc_verbose = ('Range','Azimuth','Channel'))

         s.data.t = Ndarray(BF['mtaxe'].T)
#         Ndarray(,
#                            axes = ['N'],
#                            desc = 'Time axis')
#                            desc = 'Corresponding time',
#                            shape_desc = ('N','1'))

         s.phi_min = mat_file['p'][0,0]['phi_min'][0,0]
         s.phi_max = mat_file['p'][0,0]['phi_max'][0,0]
         s.R_min = mat_file['p'][0,0]['R_min'][0,0]
         s.R_max = mat_file['p'][0,0]['R_max'][0,0]
         
         

         s.signal = Signal()
         s.signal.desc   = 'upchirp'
         s.signal.fc     = Ndarray(BF['fc' ][0, 0])     # Carrier frequency
         
         return s
         
      # Some datasets simply do not fall into a general category. We'll handle these
      # individually...
      elif origin.type == 'barge':
         

         # Using scipy.io.loadmat() to load .mat files. The result is a nparray, where structures
         # in the array are dictionaries (the data is treated later on..). 
         print 'Loading ' + PHDCODE_ROOT + origin.path + origin.file
         mat_file = io.loadmat(PHDCODE_ROOT + origin.path + origin.file)
         
         # Load the 'Sonar' dictionary and get rid of empty dimensions:
         Sonar = mat_file['Sonar'][0, 0]   # See http://projects.scipy.org/numpy/wiki/ZeroRankArray
         
         s = System()
#         s.medium = Medium()
#         s.medium.c = Sonar['c'][0, 0]   # Propagation speed
#
#         # Format: Sonar[dictionary key][hydrophone coordinates, bank]
#         rx_x = Sonar['rx_x'][:, 1]         # RX x-coordinates bank 2 (HUGIN body)
#         rx_y = Sonar['rx_y'][:, 1]         # RX y-coordinates bank 2 (HUGIN body)
#         rx_z = Sonar['rx_z'][:, 1]         # RX z-coordinates bank 2 (HUGIN body)
#         rx_yz = sqrt(rx_y ** 2 + rx_z ** 2)    # RX y-z distance (new z-axis)
#         p = Coordinate(x=rx_x, y=None, z=rx_yz)# New RX coordinates [x, axis]
##         p = vstack((rx_x, rx_yz)).T   # New RX coordinates [x, axis]
#
#         s.array = Array()
#         s.array.p = p
##         ,
##                             desc='RX coordinates',
##                             shape_desc=('x','axis'))

         s.data = Data()

         s.data.X = Ndarray(mat_file['data1'])
         
         s.signal = Signal()
         s.signal.fc = Ndarray(Sonar['fc'][0,0])
         
         s.medium = Medium()
         s.medium.c = Ndarray(Sonar['c'][0,0])    # Propagation speed
         
         
#         s.signal = Signal()
#         s.signal.fc = Sonar['fc'][0, 0]    # Carrier frequency
#         s.signal.bw = Sonar['bw'][0, 0]
#         s.signal.desc = 'upchirp'# Waveform

         return s
      
      elif origin.type == 'ultrasound multi file':  
        
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
      
      elif origin.type == 'csound':
         
         # Using scipy.io.loadmat() to load .mat files. The result is a nparray, where structures
         # in the array are dictionaries (the data is treated later <   on..). 
         print 'Loading ' + PHDCODE_ROOT + origin.path + origin.file
         mat_file = io.loadmat(PHDCODE_ROOT + origin.path + origin.file)
         
         if origin.index == 1:
         
            beamformedCycle  = mat_file['beamformedcycle'][0,0]
            #beamformedFrames = beamformedCycle[0]
            beamformedCubes = beamformedCycle[1]
                    
            NFrames   = beamformedCycle[2]
            NFrames   = NFrames[0, 0]
            NElements = beamformedCycle[3]
            NElements = NElements[0, 0]
            
            params = beamformedCycle[4]
            del beamformedCycle
            params = params[0, 0] 
            
            #delays = params[0]
            angles = params[1]*np.pi/180
            ranges = params[2]
            #phaseFactor = params[3]
            
            del params
            
            tmp = beamformedCubes[0,0].shape
            Xd = np.zeros((NFrames,tmp[0],tmp[1],tmp[2]),dtype=np.complex64)
            
            for i in range(NFrames):
               Xd[i,:,:,:] = beamformedCubes[i,0][:,:,:]
               
         elif origin.index == 2 or origin.index == 3:
            
            if origin.index == 2:
               NFrames = 10
            else:
               NFrames = 60
         
            NElements = mat_file['NElements'][0,0]
            angles = mat_file['angles']
            ranges = mat_file['ranges'].T
            
            tmp = mat_file['frame%d'%1].shape
            
            Xd = np.zeros((NFrames, tmp[0], tmp[2], tmp[1]), dtype=np.complex64)
            
            for i in range(NFrames):
               Xd[i,:,:,:] = np.transpose(mat_file['frame%d'%(i+1)], (0, 2, 1))
         
         else:
            pass
#         
         s = System()

         s.data = Data()
         s.data.Xd      = Xd
         s.data.angles  = angles
         s.data.ranges  = ranges
         s.data.M = NElements

         return s
         
      elif origin.type == 'ultrasound_simulation':
         
         import tables as tb
         s = System()
         s.data = Data()
                  # Using scipy.io.loadmat() to load .mat files. The result is a nparray, where structures
         # in the array are dictionaries (the data is treated later on..). 
         print 'Loading ' + PHDCODE_ROOT + origin.path + origin.file
         
         file = tb.openFile( (PHDCODE_ROOT + origin.path + origin.file), 'r' )
             
         root = file.getNode( file.root )

#         s.data.fc = root.frame_1.fc.read()
#         s.data.c  = root.frame_1.c.read()
         s.data.M       = root.frame_1.N_rx.read()
         s.data.angles  = root.frame_1.theta.read()[0] # Only first row makes sence to me :p 
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
      
      
#               
#               h5.createTable( group,   )
         # Open the file
#         h5 = tables.openFile(CONF_FILE, mode = "w")
#         filters = tables.Filters(complevel=5, complib='zlib') # 'zlib', 'bzip2', 'lzo'
#         hmm = h5.createCArray(h5.root, "test", tables.ComplexAtom(itemsize=16), self.conf[0]['Data']['X'].shape, filters=filters)
##         t = timeit.Timer(stmt=s)
#         hmm[:,:,:] = self.conf[0]['Data']['X']
#         h5.close()
#         
#      def load(self):
#         print 'Loading library...'
#         h5 = tables.openFile("data/configuration.lib", mode = "r")
##         filters = tables.Filters(complevel=9, complib='blosc') # 'zlib', 'bzip2', 'lzo'
##         hmm = h5.createCArray(h5.root, "test", tables.ComplexAtom(itemsize=16), self.conf[0]['Data']['X'].shape, filters=filters)
##         t = timeit.Timer(stmt=s)
#         data = h5.root.test
#         hmm = zeros(h5.root.test.shape, dtype=complex64)
#         hmm[:,:,:] = h5.root.test
#         h5.close()
#         fn = open('data/library.dat', 'rb') 
#         self.conf = pickle.load(fn)
#         fn.close
         
         
#      def show( self, *args ):
#         Ni = len(args)
#         if Ni == 0:
#            newFig()
#         elif Ni == 1:
#            newFig(args)
#         else:
#            print 'WW Too many arguments for Dataset.show()'
#         
#         set(gcf,'name','Dataset visualisation.')
#         xlabel('x [m]')
#         ylabel('z [m]')
#         title('All Windows in ''WindowList''')

#   function load(self)
#      persistent sn;
#      persistent dn;
#      
#      if( ~isempty(sn) )
#         for i=1:length(sn)
#            evalin('base',['clear ',sn{i}])
#         end
#      end
#      if( ~isempty(dn) )
#         for i=1:length(dn)
#            evalin('base',['clear ',dn{i}])
#         end
#      end
#      
#      sn = fieldnames(self.s);
#      dn = fieldnames(self.d);
#      for i=1:length(sn)
#         assignin('base',sn{i},self.s.(sn{i}))
#      end
#      for i=1:length(dn)
#         assignin('base',dn{i},self.d.(dn{i}))
#      end
#      #evalin('base',
#   end
#
#   function delayX(self,index)
#      
#   end


# /hom/dsb/projects/matlab/beamforming/functions/
