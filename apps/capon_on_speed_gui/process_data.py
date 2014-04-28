'''
Created on 7. aug. 2012

@author: jpaasen
'''
import framework.mynumpy as np

from framework.System import System
from framework.beamformer.capon import getCaponCUDA as getCaponCUDA

VERBOSE = True

class Parameter:
   
   def __init__(self, name, initValue, minValue, maxValue, increment=1, needReProcessing=True):
      self.name = name
      self.initValue = initValue
      self.value = initValue
      self.minValue = minValue
      self.maxValue = maxValue
      self.increment = increment
      self.needReProcessing = needReProcessing
      self.updated = False
   
   def incrementParam(self):
      if (self.value + self.increment <= self.maxValue):
         self.value += self.increment 
   
   def decrementParam(self):
      if (self.value - self.increment >= self.minValue):
         self.value += self.increment 
         
   def resetValue(self):
      self.value = self.initValue
      
   def mvToMax(self):
      self.value = self.maxValue
      
   def mvToMin(self):
      self.value = self.minValue
      
   def updateValue(self, v):
      if v is not self.value and v >= self.minValue and v <= self.maxValue:
         self.value = v
         self.updated = True
      
   def isUpdated(self):
      if self.updated:
         self.updated = False
         return True
      return False
      

class CaponProcessor:
   
   def __init__(self, dataTag, multiFile=False):
      
      if multiFile:
         self.s = System(dataTag + ' %d'%1)
         
         self.Xd = self.s.data.Xd
         self.Ni = self.s.data.NFrames  # No. frames
         self.Nx = self.Xd.shape[0]     # No. beams
         self.Ny = self.Xd.shape[1]     # No. range samples
         self.Nm = self.Xd.shape[2]     # No. channels
         
      else:
         self.s = System(dataTag)

         self.Xd = self.s.data.Xd
         self.Ni = self.Xd.shape[0]     # No. frames
         self.Ny = self.Xd.shape[1]     # No. range samples
         self.Nx = self.Xd.shape[2]     # No. beams
         self.Nm = self.Xd.shape[3]     # No. channels
      
      self.multiFile = multiFile
      self.dataTag = dataTag
      noReprocessing = False
      
#      self.sliceRangeStart    = Parameter('Range slice start',    0, 1, self.Ny)
#      self.sliceAngleStart   = Parameter('Angle slice start',    0, 1, self.Nx)
#      self.sliceElementStart = Parameter('Element slice start',  0, 1, self.Nm)
#      self.sliceRangeEnd      = Parameter('Range slice end',     self.Ny, 1, self.Ny)
#      self.sliceAngleEnd     = Parameter('Angle slice end',     self.Nx, 1, self.Nx)
#      self.sliceElementEnd   = Parameter('Element slice end',   self.Nm, 1, self.Nm)
#      self.sliceRangeStep     = Parameter('Range slice step', 1, 1, self.Ny/2)
#      self.sliceAngleStep    = Parameter('Angle slice step', 1, 1, self.Nx/2)
#      self.sliceElementStep   = Parameter('Element slice step', 1, 1, self.Nm/2)

      self.sliceRangeStart    = Parameter('Range slice start',    self.Ny/4, 1, self.Ny)
      self.sliceAngleStart   = Parameter('Angle slice start',    self.Nx/5, 1, self.Nx)
      self.sliceElementStart = Parameter('Element slice start',  0, 1, self.Nm)
      self.sliceRangeEnd      = Parameter('Range slice end',     self.Ny*3/4, 1, self.Ny)
      self.sliceAngleEnd     = Parameter('Angle slice end',     self.Nx*4/5, 1, self.Nx)
      self.sliceElementEnd   = Parameter('Element slice end',   self.Nm, 1, self.Nm)
      self.sliceRangeStep     = Parameter('Range slice step', 1, 1, self.Ny/2)
      self.sliceAngleStep    = Parameter('Angle slice step', 1, 1, self.Nx/2)
      self.sliceElementStep   = Parameter('Element slice step', 1, 1, self.Nm/2)
      
      self.Ky = Parameter('Radial upsampling', 4, 1, 8, 1, noReprocessing) # Up-sampling factors 
      self.Kx = Parameter('Lateral upsampling', 2, 1, 8, 1, noReprocessing)
      self.interp_method = 0 # 0 imag and real interp, 1 angle and radian interp, 2 fft2 interp
      
      self.Nb = Parameter('Beamspace dimension', 0, 0, self.Nm) # Dim. beamspace
    
      # Inital Capon parameters
      self.d = Parameter('Diagonal loading (power of 10)', -1, -5, 5)     # Diagonal loading in percent
      self.L = Parameter('Subarray size', int(self.Nm/2), 1, self.Nm)  # Subarray size <= Nm/2 (e.g. 16)
      self.K = Parameter('Time averaging', 2, 0, (self.Ny-1)/2) # Capon range-average window (+- this value)
      self.B = np.array([0]) # Subspace matrix

      self.image_idx = Parameter('Frame no.', 0, 0, self.Ni-1)
      
      self.minDynRange = Parameter('Min dynamic range', -20, -60, 60, 1, noReprocessing)
      self.maxDynRange = Parameter('Max dynamic range', 20, -60, 60, 1, noReprocessing)
      self.minDynRangeCapon = Parameter('Min dynamic range Capon', -20, -60, 60, 1, noReprocessing)
      self.maxDynRangeCapon = Parameter('Max dynamic range Capon', 20, -60, 60, 1, noReprocessing)
      
      self.fps = Parameter('Video fps', 10, 1, 50, 1, noReprocessing) # TODO: Move to gui
      
#      self.profile_on_off = Parameter('Profile plots on (1) off (0)', 1, 0, 1, 1, noReprocessing)
      self.save_single_plots = Parameter('Save single plots: on=1, off=0', 0, 0, 1, 1, noReprocessing) # TODO: Move to gui
      
      self.show_legends = Parameter('Show legends: on=1, off=0', 0, 0, 1, 1, noReprocessing)
      
      self.apod = Parameter('Apodization: 0=uniform, 1=hamming', 0, 0, 1)
      
      #self.compress = Parameter('Number of logs: 1=log(), 2=log(log())', 1, 1, 2, 1, noReprocessing)
      
      # profile position (Lateral and axial pattern is plotted, crossing at this position)
      self.profilePos = [10.0, 80.0, 0.0]
      
   def calcBeamPatternLinearArray(self, w, M, spacing, thetas, steering_angle=0, takePowerAndNormalize=False):
      x_el = np.linspace( -(M-1)/(2.0/spacing), (M-1)/(2.0/spacing), M )
      W_matrix = np.exp(-1j * np.outer(2*np.pi*(np.sin(thetas) - np.sin(steering_angle)), x_el))
      W = np.dot(W_matrix , w)
      if takePowerAndNormalize:
         W = W*W.conj()
         W = W / W.max()
      
      return W
   
   def logCompress(self, img, min, max):
      img_log_norm = np.db(abs(img)) - np.db(np.mean(abs(img)))
      #img_clip = np.clip(img_log_norm, min, max)
      return img_log_norm
   
   def ampCompress(self, img, min, max):
      # todo should map amplitude values using err/sigmoid function
      #img_log = self.logCompress(img, min, max)
      #img_mapped = (img_log - min)/(max-min)  
      return img#np.log10(9*img_mapped + 1)

   def processData(self):
   
      if VERBOSE:
         print 'Start processing image...'
   
      if self.multiFile:
         self.s = System(self.dataTag + ' %d'%(self.image_idx.value+1))
         self.Xd_i = self.s.data.Xd
      else:
         self.Xd_i = self.Xd[self.image_idx.value, :, :, :].copy()
         
      self.Xd_i = self.Xd_i[self.sliceAngleStart.value:self.sliceAngleEnd.value:self.sliceAngleStep.value, 
                            self.sliceRangeStart.value:self.sliceRangeEnd.value:self.sliceRangeStep.value, 
                            self.sliceElementStart.value:self.sliceElementEnd.value:self.sliceElementStep.value];
      self.Nx = self.Xd_i.shape[0]     # No. beams
      self.Ny = self.Xd_i.shape[1]     # No. range samples
      self.Nm = self.Xd_i.shape[2]     # No. channels
      
      self.angles = self.s.data.angles.squeeze()
      self.angles = self.angles[self.sliceAngleStart.value:self.sliceAngleEnd.value:self.sliceAngleStep.value]
      self.ranges = self.s.data.ranges.squeeze()
      self.ranges = self.ranges[self.sliceRangeStart.value:self.sliceRangeEnd.value:self.sliceRangeStep.value]
   
      # Make delay and sum image
      if self.apod.value is 1:
         self.das_w = np.hamming(self.Nm)
         self.das_w_sub = np.hamming(self.L.value)
      else:
         self.das_w = np.ones(self.Nm)
         self.das_w_sub = np.ones(self.L.value)
      
      self.img_das = np.dot(self.Xd_i, self.das_w) / self.Nm
      if self.K.value > 0:
         self.img_das = self.img_das[:,self.K.value:-self.K.value]# truncate to get equal dim on das and capon image
         
      # init sub/beam-space matrix
      if self.Nb.value > 0:
         self.B = np.ones((self.Nb.value, self.L.value))
      else:
         self.B = np.array([0])

      # Make capon image
      res_gpu = getCaponCUDA.getCaponCUDAPy(self.Xd_i, 10.0**self.d.value, self.L.value, self.K.value, self.B, False, False)
      self.img_capon = res_gpu[0]
      self.capon_weights = res_gpu[2]
      #self.img_capon = self.img_das
      
      if VERBOSE:
         print 'getCaponCUDA return code: ', res_gpu[3]
         print 'done.'
         
   def interpolateData(self):
      import scipy.interpolate as interpolate 
      
      NyK = self.Ny - 2*self.K.value # The Capon image will be 2*K smaller in range

      if VERBOSE:
         print 'Start interpolating...'

      # IQ-interpolation (in image domain, imag and real, hence coherent) 
      iq_interp_factor = 2
      x_idx = np.arange(self.Nx)
      y_idx = np.arange(NyK)
      x_up_idx = np.linspace(0.25, self.Nx-1-0.25, (self.Nx-1)*iq_interp_factor)  
      y_up_idx = np.arange(NyK)
      self.angles_intrp = interpolate.interp1d( x_idx, self.angles ) (x_up_idx)
      self.ranges_intrp = self.ranges
      if self.K.value > 0:
         self.ranges_intrp = self.ranges_intrp[self.K.value:-self.K.value]
          
      img_das_real = interpolate.RectBivariateSpline( x_idx, y_idx, self.img_das.real ) (x_up_idx, y_up_idx)
      img_das_imag = interpolate.RectBivariateSpline( x_idx, y_idx, self.img_das.imag ) (x_up_idx, y_up_idx)
   
      img_capon_real = interpolate.RectBivariateSpline( x_idx, y_idx, self.img_capon.real ) (x_up_idx, y_up_idx)
      img_capon_imag = interpolate.RectBivariateSpline( x_idx, y_idx, self.img_capon.imag ) (x_up_idx, y_up_idx)
      
      self.img_das_iq_intrp   = img_das_real   + 1j * img_das_imag
      self.img_capon_iq_intrp = img_capon_real + 1j * img_capon_imag   
      
      # In-coherent interpolation
      if self.Ky.value > 1 or self.Kx.value > 1:
         
         NxK = (self.Nx-1) * iq_interp_factor
      
         y_idx = np.arange(NyK)
         x_idx = np.arange(NxK)
      
         y_up_idx = np.linspace(0,NyK-1,NyK*self.Ky.value)
         x_up_idx = np.linspace(0,NxK-1,NxK*self.Kx.value)
         
         self.angles_intrp = interpolate.interp1d( x_idx, self.angles_intrp ) (x_up_idx)
         if self.K.value > 0:
            self.ranges_intrp = interpolate.interp1d( y_idx, self.ranges[self.K.value:-self.K.value].squeeze() ) (y_up_idx)
         else:
            self.ranges_intrp = interpolate.interp1d( y_idx, self.ranges ) (y_up_idx)
      
         self.img_das_intrp   = interpolate.RectBivariateSpline( x_idx, y_idx, abs(self.img_das_iq_intrp) ) (x_up_idx, y_up_idx)
         self.img_capon_intrp = interpolate.RectBivariateSpline( x_idx, y_idx, abs(self.img_capon_iq_intrp) ) (x_up_idx, y_up_idx)
            
      else: # do nothing
         self.img_das_intrp = self.img_das_iq_intrp
         self.img_capon_intrp = self.img_capon_iq_intrp
         
      self.img_das_intrp   = np.transpose(self.img_das_intrp)
      self.img_capon_intrp = np.transpose(self.img_capon_intrp)
      
      self.img_das_detected = self.logCompress(self.img_das_intrp,   self.minDynRange.value, self.maxDynRange.value)
      self.img_cap_detected = self.logCompress(self.img_capon_intrp, self.minDynRangeCapon.value, self.maxDynRangeCapon.value)
         
      if VERBOSE:
         print 'done'
         
   def scannConvertImage(self):
      pass
      #I_new = np.array()
      #for x=1,...,width:
      #   for y=1,...,height:
      #      angle=atan2(y, x)
      #      r=sqrt(x^2+y^2)
      #      I_new(x, y)=I(angle, r)

   def plot(self, axis1, axis2, axis3=None, axis4=None):
      # Display results
      import framework.mypylab as pl
      from framework.mynumpy import abs, db, mean, sin, cos, pi

      self.interpolateData()

      if VERBOSE:
         print 'Start plotting...'

      theta,rad = np.meshgrid(self.angles_intrp, self.ranges_intrp)
      x = 1000 * rad * sin(theta)
      y = 1000 * rad * cos(theta)
      
      ## Start Plotting
      if (axis1 == 0): # stand alone plotting
         pl.figure()
         pl.subplot(1,2,1, aspect=1)
         pl.pcolormesh(x, y, self.img_das_detected, cmap=pl.cm.gray, vmin=self.minDynRange.value, vmax=self.maxDynRange.value)
         pl.gca().invert_yaxis()
      else: # Plotting in given axis
         axis1.pcolormesh(x, y, self.img_das_detected, cmap=pl.cm.gray, vmin=self.minDynRange.value, vmax=self.maxDynRange.value)
         axis1.set_title('Delay-and-sum', fontsize='large')
         axis1.set_xlabel('Width [mm]', fontsize='large')
         axis1.set_ylabel('Depth [mm]', fontsize='large')
         axis1.set_xlim(x.min(), x.max())
         axis1.set_ylim(y.max(), y.min())
      
      if (axis2 == 0):
         ax = 0
         pl.subplot(1,2,2, aspect=1)
         pl.pcolormesh(x, y, self.img_cap_detected, cmap=pl.cm.gray, vmin=self.minDynRangeCapon.value, vmax=self.maxDynRangeCapon.value)
         pl.gca().invert_yaxis()
         pl.show()
      else:
         ax = axis2.pcolormesh(x, y, self.img_cap_detected, cmap=pl.cm.gray, vmin=self.minDynRangeCapon.value, vmax=self.maxDynRangeCapon.value)
         if self.Nb.value > 0:
            axis2.set_title('BS-Capon', fontsize='large')
         else:
            axis2.set_title('ES-Capon', fontsize='large')
         axis2.set_xlabel('Width [mm]', fontsize='large')
         axis2.set_ylabel('Depth [mm]', fontsize='large')
         axis2.set_xlim(x.min(), x.max())
         axis2.set_ylim(y.max(), y.min())
         
      if axis3 is not None:
         # plot axial profile
         #profile_angle = np.arctan( self.profilePos[0] / self.profilePos[1] )
         #if profile_angle < self.angles_intrp[-1] and profile_angle > self.angles_intrp[0]: 
         #   range_slice_idx = round(self.angles_intrp.shape[0] * (profile_angle-self.angles_intrp[0]) / (self.angles_intrp[-1]-self.angles_intrp[0]))
         #   img_das_rslice = img_das_detected[:, range_slice_idx]
         #   img_cap_rslice = img_cap_detected[:, range_slice_idx]
         #
         #   axis3.plot(y[:,range_slice_idx], img_das_rslice, label='DAS')
         #   axis3.plot(y[:,range_slice_idx], img_cap_rslice, '-r', label='Capon')
         #   
         #   axis3.set_ylim([self.minDynRange.value, self.maxDynRange.value])
         #   
         #   axis3.set_title('Radial intensity at %d degrees'%round(profile_angle*180/np.pi))
         #   axis3.set_xlabel('Depth [mm]')
         #   axis3.set_ylabel('Radial intensity [dB]')
         #   axis3.legend(loc=3, markerscale=0.5)
         
         # Plot beampatterns and power spectrums
         profile_range = np.sqrt( self.profilePos[0]**2 + self.profilePos[1]**2 ) / 1000.0
         profile_angle = np.arctan( self.profilePos[0] / self.profilePos[1] )
         
         if profile_range < self.ranges_intrp[-1] and profile_range > self.ranges_intrp[0] and profile_angle < self.angles_intrp[-1] and profile_angle > self.angles_intrp[0]:
            
            range = round(self.ranges_intrp.shape[0] * (profile_range-self.ranges_intrp[0]) / (self.ranges_intrp[-1]-self.ranges_intrp[0]))
            angle = round(self.angles_intrp.shape[0] * (profile_angle-self.angles_intrp[0]) / (self.angles_intrp[-1]-self.angles_intrp[0]))
            
            spacing = 0.5
            
            # plot beamspace data
            x_data = self.Xd_i[angle / (2*self.Kx.value), range / (1*self.Ky.value), :]
            das_beams = self.calcBeamPatternLinearArray(x_data, self.Nm, spacing, self.angles_intrp, self.angles_intrp[angle], takePowerAndNormalize=True)
            axis3.plot(x[range,:], 10*np.log10(das_beams) + self.maxDynRange.value, label='DAS Sample Spectrum')
            
            # make Capon beam pattern
            w_capon = self.capon_weights[angle / (2*self.Kx.value), range / (1*self.Ky.value), :]
            W_capon = self.calcBeamPatternLinearArray(w_capon, self.L.value, spacing, self.angles_intrp, self.angles_intrp[angle])
            W_capon = abs(W_capon*W_capon.conj())
            W_capon = W_capon / W_capon[angle]
            axis3.plot(x[range,:], 10*np.log10(W_capon) + self.maxDynRange.value - 10 , label='Capon Beampattern')
            
            # make DAS beam pattern for subarrays
            W_das = self.calcBeamPatternLinearArray(self.das_w_sub, self.L.value, spacing, self.angles_intrp, self.angles_intrp[angle], takePowerAndNormalize=True)       
            axis3.plot(x[range,:], 10*np.log10(W_das) + self.maxDynRange.value - 10, label='DAS Beampattern')
            
            # make total Capon beam pattern for the whole system (tx and rx)?
            
            # plot selected direction/angle
            axis3.plot([x[range,angle], x[range,angle]], [self.minDynRange.value, self.maxDynRange.value], label='Steering angle')
         
         axis3.set_ylim([self.minDynRange.value, self.maxDynRange.value])
         axis3.set_title('Beam pattern at %d degrees and %d mm range'%(round(profile_angle*180/np.pi), round(profile_range*1000)), fontsize='large') 
         axis3.set_xlabel('Width [mm]', fontsize='large')
         axis3.set_ylabel('Intensity/Gain [dB]', fontsize='large')
         if self.show_legends.value is 1:
            axis3.legend(loc=3, markerscale=0.5)
      
      if axis4 is not None:
         profile_range = np.sqrt(self.profilePos[0]**2 + self.profilePos[1]**2) / 1000.0
         
         if profile_range < self.ranges_intrp[-1] and profile_range > self.ranges_intrp[0]:
            angle_slice_idx = round(self.ranges_intrp.shape[0] * (profile_range-self.ranges_intrp[0]) / (self.ranges_intrp[-1]-self.ranges_intrp[0]))
            self.img_das_aslice = self.img_das_detected[angle_slice_idx, :]
            self.img_cap_aslice = self.img_cap_detected[angle_slice_idx, :]
            self.x_aslice = x[angle_slice_idx,:]
         
            axis4.plot(self.x_aslice, self.img_das_aslice, label='DAS')
            if (self.Nb.value > 0):
               axis4.plot(self.x_aslice, self.img_cap_aslice, '-r', label='BS-Capon')
            else:
               axis4.plot(self.x_aslice, self.img_cap_aslice, '-r', label='ES-Capon')
            
            axis4.set_ylim([self.minDynRange.value, self.maxDynRange.value])
            
            axis4.set_title('Lateral intensity at %d mm range'%round(profile_range*1000), fontsize='large')
            axis4.set_xlabel('Width [mm]', fontsize='large')
            axis4.set_ylabel('Lateral intensity [dB]', fontsize='large')
            if self.show_legends.value is 1:
               axis4.legend(loc=3, markerscale=0.5)
        
      if VERBOSE:   
         print 'done'
         
      return ax

if __name__ == '__main__':
   
   beamformers = CaponProcessor('motion phantom 16x 3.4 MHz 20dB30dB')
   beamformers.processData()
   beamformers.plot(0,0)
