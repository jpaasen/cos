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

   #HISAS 1030 Speckle Realisations:
   s = FileRegister()
   s.desc   = 'HISAS 1030 Cube: Speckle Realisation'
   s.type   = 'hisas_sim'
   s.group = 'hisas_speckle'
   s.index = 1
   s.date   = '2010.01.29'
   s.file   = ''
   s.lpath = '/data/sonar/simulations/hisas1030/speckle_cube/'
   s.rhost = 'login.ifi.uio.no'
   s.rpath = '/projects/safiye/Andreas/DataCubes/Realizations/'
   si = FileRegister(s)
   
   for i in range(1,101):
      si.index = i
      si.desc   = s.desc + ' ' + "%d"%i
      si.file   = 'Specle_' + "%d"%i + '_Cube.mat'
      data.append(FileRegister(si))


   # Homengraa wreck:
   s.desc   =   'HUGIN with HISAS 1030: Holmengraa Wreck'
   s.type   =   'hugin_delayed'
   s.group =   'holmengraa'
   s.index =    1
   s.date   =   '2010.01.29'
   s.file   =   'Holmengraa.mat'
   s.lpath =   '/data/sonar/real/hugin/hisas1030/'
   s.rhost =   'login.ifi.uio.no'
   s.rpath =   '/projects/safiye/Andreas/DataCubes/Vrak/'
   data.append(FileRegister(s))

   # Barge wreck:
   s.desc   =   'HUGIN with HISAS 1030: Barge'
   s.type   =   'focus_dump_type_B'
   s.group =   'barge'
   s.index =    1
   s.date   =   '2009.23.01'
   s.file   =   'barge.mat'
   s.lpath =   '/data/sonar/real/hugin/hisas1030/'
   s.rhost =   'login.ifi.uio.no'
   s.rpath =   '/projects/safiye/Data/measurements/'
   data.append(FileRegister(s))

   # The Byfjorden wreck:
   s.desc   = 'HUGIN with HISAS 1030: Byfjorden wreck'
   s.type   = 'focus_dump_type_A'
   s.group = 'byfjorden'
   s.index =   1
   s.date   = '2008.08.27'
   s.file   = 'run080827_1-l04-s0b2p1300.mat'
   s.lpath = '/data/sonar/real/hugin/hisas1030/'
   s.rhost = 'ffi'
   s.rpath = '/not/available/'
   data.append(FileRegister(s))
   
   ## SAS corrected data ##
   s.desc   = 'HUGIN with HISAS 1030: Cross A - MF parameters'
   s.type   = 'focus_sas_dump_type_A_parameters'
   s.group = 'crossA_parameters'
   s.index =   1
   s.date   = '2012.08.22'
   s.file   = 'mf_params.mat'
   s.lpath = '/data/sonar/real/hugin/hisas1030/run_20100624_01_l01-1_cross/'
   s.rhost = 'FOCUS machine'
   s.rpath = '...'
   data.append(FileRegister(s))
   
     
   s.desc   = 'HUGIN with HISAS 1030: Cross A'
   s.type   = 'focus_sas_dump_type_A_data'
   s.group  = 'crossA_data'
   s.index  =   1
   s.date   = '2012.08.22'
   s.file   = ''
   s.lpath = '/data/sonar/real/hugin/hisas1030/run_20100624_01_l01-1_cross/'
   s.rhost = 'FOCUS machine'
   s.rpath = '...'
   si = FileRegister(s)
   
   for i in range(0,21):
      si.index = i
      si.desc   = s.desc + ' - ping ' + "%d"%(i+420)
      si.file   = 'mfping%d_s0b2.mat'%(i+420)
      data.append(FileRegister(si))
      
   ## SSS dump from the sectorscan dump script on the FOCUS machine ##
   s.desc   = 'HUGIN with HISAS 1030: Cross A - MF parameters'
   s.type   = 'focus_sss_dump_type_A'
   s.group = 'holmengraa_sss_oversampled'
   s.index =   1
   s.date   = '2012.08.23'
   s.file   = 'run120202_4 h_graa1-1.mat'
#   s.file   = 'run_20100624_01_l01_1.mat'
   s.lpath = '/data/sonar/real/hugin/hisas1030/run120202_4 h_graa1-1/'
   s.rhost = 'FOCUS machine'
   s.rpath = '...'
   data.append(FileRegister(s))
   
   ## Below here is ultrasound datasets
   
   # Vingmed experimental channel data:
   s.desc   = 'Vingmed experimental channel data'
   s.type   = 'csound'
   s.group = 'csound'
   s.index =   1
   s.date   = '2011'
   s.file   = 'us_channel_data_vingmed_4frames.mat'
   s.lpath = '/data/ultrasound/real/vingmed/'
   s.rhost = 'vingmed'
   s.rpath = '/not/available/'
   data.append(FileRegister(s))
   
   # Vingmed invivo liver
   s.desc   = 'Vingmed liver channel data'
   s.type   = 'csound'
   s.group = 'csound'
   s.index =   2
   s.date   = '2011'
   s.file   = '20120323_Liver_dump6_10frames_50tx_2MLA.mat'
   s.lpath = '/data/ultrasound/real/vingmed/'
   s.rhost = 'vingmed'
   s.rpath = '/not/available/'
   data.append(FileRegister(s))
   
   # Vingmed experimental channel data: # TODO: Change to be specified by multiple files
   s.desc   = 'Vingmed experimental channel data'
   s.type   = 'csound'
   s.group = 'csound'
   s.index =   3
   s.date   = '2011'
   s.file   = '20120323_CardiacBmode53fps_dump1_60frames_50tx_1MLA.mat'
   s.lpath = '/data/ultrasound/real/vingmed/'
   s.rhost = 'vingmed'
   s.rpath = '/not/available/'
   data.append(FileRegister(s))
   
   # Cardiac dataset with 2STB
   addUsMultiFileDataset(data, 60, 'Vingmed experimental channel data', 'csound cardiac 2STB 1', '2011', 
                         '20120323_CardiacBmode53fps_dump1_60frames_50tx_1MLA_2STBchannel_', 
                         'real/vingmed/20120323_CardiacBmode53fps_dump1_60frames_50tx_1MLA_2STBchannel/', 
                         'vingmed')
   
   # Cardiac dataset with 4MLA
   addUsMultiFileDataset(data, 60, 'Vingmed experimental channel data', 'csound cardiac 4MLA 1', '2011', 
                         '20120323_CardiacBmode53fps_dump1_60frames_50tx_4MLA_', 
                         'real/vingmed/20120323_CardiacBmode53fps_dump1_60frames_50tx_4MLA/', 
                         'vingmed')
   
   # Liver datasets with 2STB
   addUsMultiFileDataset(data, 10, 'Vingmed experimental channel data', 'csound liver 2STB 1', '2011', 
                         '20120323_Liver_dump4_10frames_50tx_1MLA_2STBchannel_', 
                         'real/vingmed/20120323_Liver_dump4_10frames_50tx_1MLA_2STBchannel/', 
                         'vingmed')
   addUsMultiFileDataset(data, 10, 'Vingmed experimental channel data', 'csound liver 2STB 2', '2011', 
                         '20120323_Liver_dump7_10frames_50tx_1MLA_2STBchannel_', 
                         'real/vingmed/20120323_Liver_dump7_10frames_50tx_1MLA_2STBchannel/', 
                         'vingmed')
   
   # simulation of motion phantom 16x oversampling 2.5 MHz
   addUsMultiFileDataset(data, 50, 'Field II simulation', 'motion phantom 16x', '2011', 
                         'motion_phantom_16x_', 
                         'simulations/motion_phantom_16x/', 
                         'Field II')
   
   # simulation of motion phantom 16x oversampling
   addUsMultiFileDataset(data, 50, 'Field II simulation', 'motion phantom 16x 3.4 MHz', '2013', 
                         'motion_phantom_16x_', 
                         'simulations/motion_phantom_16x_3point4MHz/', 
                         'Field II')
   
   addUsMultiFileDataset(data, 50, 'Field II simulation', 'motion phantom 16x 3.4 MHz 20dB30dB', '2013', 
                         'motion_phantom_16x_', 
                         'motion_phantom_16x_3-4MHz_20dB30dB/', 
                         'Field II')   
   
   # Add more ultrasound recordings

   # Crazy JP simulation
   s.desc   = 'Ultrasound simulation 1'
   s.type   = 'ultrasound_simulation'
   s.group = 'ultrasound_simulation'
   s.index =   1
   s.date   = '2011'
   s.file   = 'phantom_data.h5'
   s.lpath = '/data/ultrasound/simulations/'
   s.rhost = 'vingmed'
   s.rpath = '/not/available/'
   data.append(FileRegister(s))
   
   # Add oversampled phantom
   
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
   
   

#% /hom/dsb/projects/matlab/beamforming/functions/
