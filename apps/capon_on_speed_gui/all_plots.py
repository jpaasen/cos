
import framework.mynumpy as np
from framework.mynumpy import abs, sin, cos, sqrt, pi
import framework.mypylab as pl
from process_vingmed_data import CaponProcessor  

# controlling diagonal loading for ES and BS Capon
d_ES = -2
d_BS = -2

def process_and_plot_phantom_image(Nb=0):
   
   datatag = 'motion phantom 16x 3.4 MHz 20dB30dB'
   #datatag.append('Vingmed data liver 2STB 1')
   #datatag.append('Vingmed data liver 2STB 2')
   #datatag.append('Vingmed data cardiac 2STB 1')
   #datatag.append('Vingmed data cardiac 4MLA 1')
      
   multiFile = True
        
   capon = CaponProcessor(datatag, multiFile)
   
   # set parameters for capon processor
   capon.image_idx.updateValue(15)
   
   capon.sliceRangeStart.updateValue(48)
   capon.sliceAngleStart.updateValue(195)
   capon.sliceRangeEnd.updateValue(144)
   capon.sliceAngleEnd.updateValue(780)
   
   capon.Ky.updateValue(6) # incoherent interpolation in range
   capon.Kx.updateValue(2) # in azimuth
   
   capon.Nb.updateValue(Nb)
   if capon.Nb.value > 0:
      capon.d.updateValue(d_BS)
   else:
      capon.d.updateValue(d_ES)
   
   #capon.L.updateValue()
   capon.K.updateValue(2)
   
   capon.minDynRange.updateValue(-20)
   capon.maxDynRange.updateValue(20)
   capon.minDynRangeCapon.updateValue(-20)
   capon.maxDynRangeCapon.updateValue(20)
   
   capon.show_legends.updateValue(1)
   capon.apod.updateValue(0) #'Apodization: 0=uniform, 1=hamming'
   
   # position profile
   capon.profilePos = [0.0, 93.0, 0.0]
   
   capon.processData()
   
   fig = []
   ax = []
   filename = []
   if Nb == 0:
      filename = ['das', 'capon', 'ax1', 'ax2']
   else:
      filename = ['das', 'capon_bs', 'ax1_bs', 'ax2_bs']
   
   for i in range(4):
      figure = pl.figure()
      ax.append(figure.add_subplot(1,1,1, ))
      fig.append(figure)       
   
   axc = capon.plot(ax[0],ax[1],ax[2], ax[3])
   
   #for i in range(2):
      #ax[i].invert_yaxis()
      #fig[i].tight_layout()
   
   from datetime import datetime
   d = datetime.time(datetime.now())
   for i in range(4):
      
      if i is 1 and Nb is not 0:
         colorBarFig = pl.figure()
         cbar = colorBarFig.colorbar(axc, orientation='vertical', ticks=[capon.minDynRange.value, 0, capon.maxDynRange.value], shrink=0.5)
         cbar.ax.set_yticklabels([r'$\leq$%d dB'%capon.minDynRange.value, '0 dB', r'$\geq$%d dB'%capon.maxDynRange.value])
         colorBarFig.savefig('colorBarPhantom.png', bbox_inches='tight', pad_inches=0.5, dpi=300)
         
      if i < 2:
         ax[i].set_aspect('equal')#, 'datalim'
      
      #fig[i].savefig('%s_axes%d_%d%d%d%d'%(filename, i, d.hour, d.minute, d.second, d.microsecond), bbox_inches='tight', pad_inches=0.5)
      #fig[i].savefig('%s_axes%d_%d%d%d%d.png'%(filename, i, d.hour, d.minute, d.second, d.microsecond), dpi=300, bbox_inches='tight', pad_inches=0.5)
      fig[i].savefig('%s.png'%(filename[i]), dpi=300, bbox_inches='tight', pad_inches=0.5)
   
      
def process_and_plot_invivo_image(Nb=0):
   
   datatag = 'Vingmed data cardiac 2STB 1'
      
   multiFile = True
        
   capon = CaponProcessor(datatag, multiFile)
   
   # set parameters for capon processor
   #capon.image_idx.updateValue(15)
   
   capon.sliceRangeStart.mvToMin()
   capon.sliceAngleStart.mvToMin()
   capon.sliceRangeEnd.mvToMax()
   capon.sliceAngleEnd.mvToMax()
   
   capon.Ky.updateValue(2) # incoherent interpolation in range
   capon.Kx.updateValue(1) # in azimuth
   
   capon.Nb.updateValue(Nb)
   if capon.Nb.value > 0:
      capon.d.updateValue(d_BS)
   else:
      capon.d.updateValue(d_ES)
   
   #capon.L.updateValue()
   capon.K.updateValue(2)
   
   capon.minDynRange.updateValue(-20)
   capon.maxDynRange.updateValue(20)
   capon.minDynRangeCapon.updateValue(-20)
   capon.maxDynRangeCapon.updateValue(20)
   
   capon.show_legends.updateValue(1)
   capon.apod.updateValue(0) #'Apodization: 0=uniform, 1=hamming'
   
   # position profile
   capon.profilePos = [-13.83174473246271, 96.50653026882506, 0.0]
   
   capon.processData()
   
   fig = []
   ax = []
   filename = []
   if Nb == 0:
      filename = ['das_invivo', 'capon_invivo', 'ax1_invivo', 'ax2_invivo']
   else:
      filename = ['das_invivo', 'capon_bs_invivo', 'ax1_bs_invivo', 'ax2_bs_invivo']
   for i in range(4):
      figure = pl.figure()
      ax.append(figure.add_subplot(1,1,1))
      fig.append(figure)       
   
   axc = capon.plot(ax[0],ax[1],ax[2], ax[3])
   
   #for i in range(2):
      #ax[i].invert_yaxis()
      #fig[i].tight_layout()
   
   from datetime import datetime
   d = datetime.time(datetime.now())
   for i in range(4):

      if i is 1 and Nb is not 0:
         colorBarFig = pl.figure()
         cbar = colorBarFig.colorbar(axc, ticks=[capon.minDynRange.value, 0, capon.maxDynRange.value], orientation='vertical')#, shrink=0.5)
         cbar.ax.set_yticklabels([r'$\leq$%d dB'%capon.minDynRange.value, '0 dB', r'$\geq$%d dB'%capon.maxDynRange.value])
         colorBarFig.savefig('colorBarInvivo.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
         
      if i < 2:
         ax[i].set_aspect('equal')#, 'datalim')

      #fig[i].savefig('%s_axes%d_%d%d%d%d'%(filename, i, d.hour, d.minute, d.second, d.microsecond), bbox_inches='tight', pad_inches=0.5)
      fig[i].savefig('%s.png'%(filename[i]), dpi=300, bbox_inches='tight', pad_inches=0.5)
      
      
def plot_slices_simulation(legendOn=True):
   datatag = 'motion phantom 16x 3.4 MHz 20dB30dB'
   #datatag.append('Vingmed data liver 2STB 1')
   #datatag.append('Vingmed data liver 2STB 2')
   #datatag.append('Vingmed data cardiac 2STB 1')
   #datatag.append('Vingmed data cardiac 4MLA 1')
      
   multiFile = True
  
   capon = CaponProcessor(datatag, multiFile)
   
   # set parameters for capon processor
   capon.image_idx.updateValue(15)
   
   capon.sliceRangeStart.updateValue(48)
   capon.sliceAngleStart.updateValue(295)
   capon.sliceRangeEnd.updateValue(144)
   capon.sliceAngleEnd.updateValue(680)
   
   capon.Ky.updateValue(2) # incoherent interpolation in range
   capon.Kx.updateValue(6) # in azimuth
   
   #capon.L.updateValue()
   capon.K.updateValue(2)
   
   capon.minDynRange.updateValue(-20)
   capon.maxDynRange.updateValue(25)
   capon.minDynRangeCapon.updateValue(-20)
   capon.maxDynRangeCapon.updateValue(25)
   
   capon.show_legends.updateValue(1)
   capon.apod.updateValue(0) #'Apodization: 0=uniform, 1=hamming'
   
   # position profile
   capon.profilePos = [0.0, 90.0+0.1388, 0.0]#[0.0, 93.0, 0.0]
   
   ax = []
   for i in range(4):
      figure = pl.figure()
      ax.append(figure.add_subplot(1,1,1))     
   
   Nbs = [0, 3]
   
   for Nb in Nbs:
   
      capon.Nb.updateValue(Nb)
      if capon.Nb.value > 0:
         capon.d.updateValue(d_BS)
      else:
         capon.d.updateValue(d_ES)
   
      capon.processData()
      
      capon.plot(ax[0],ax[1],ax[2], ax[3])
      
      if Nb == 0:
         x = capon.x_aslice
         das_profile = capon.img_das_aslice
         capon_profile = capon.img_cap_aslice
      else:
         capon_bs_profile = capon.img_cap_aslice
         
   fig = pl.figure()
   ax = fig.add_subplot(1,1,1)
            
   ax.plot(x, das_profile, c=(0.25,0.25,1.00), ls=':', lw=2.0, label='DAS')
   ax.plot(x, capon_profile, c=(1.00,0.25,0.25), ls='-.', lw=2.0, label='ES-Capon')
   ax.plot(x, capon_bs_profile, c=(0.0,0.0,0.0), ls='-', lw=2.0, label='BS-Capon')
   
   ax.set_ylim([capon.minDynRange.value, capon.maxDynRange.value])
   
   ax.set_title('Lateral intensity at %d mm range'%capon.profilePos[1], fontsize='large')
   ax.set_xlabel('Width [mm]', fontsize='large')
   ax.set_ylabel('Lateral intensity [dB]', fontsize='large')
   if legendOn:
      ax.legend(loc=1, markerscale=0.5)
   
   #fig.savefig('simulation_slice', bbox_inches='tight', pad_inches=0.5)
   fig.savefig('simulation_slice.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
   
def plot_slices_invivo(legendOn=True):
   datatag = 'Vingmed data cardiac 2STB 1'
   #datatag = 'motion phantom 16x'
   #datatag.append('Vingmed data liver 2STB 1')
   #datatag.append('Vingmed data liver 2STB 2')
   #datatag.append('Vingmed data cardiac 2STB 1')
   #datatag.append('Vingmed data cardiac 4MLA 1')
      
   multiFile = True
     
   capon = CaponProcessor(datatag, multiFile)
   
   capon.sliceRangeStart.mvToMin()
   capon.sliceAngleStart.mvToMin()
   capon.sliceRangeEnd.mvToMax()
   capon.sliceAngleEnd.mvToMax()
   
   # set parameters for capon processor
   capon.image_idx.updateValue(0)
   
   capon.Ky.updateValue(2) # incoherent interpolation in range
   capon.Kx.updateValue(1) # in azimuth
   
   #capon.L.updateValue()
   capon.K.updateValue(2)
   
   capon.minDynRange.updateValue(-20)
   capon.maxDynRange.updateValue(20)
   capon.minDynRangeCapon.updateValue(-20)
   capon.maxDynRangeCapon.updateValue(20)
   
   capon.show_legends.updateValue(1)
   capon.apod.updateValue(0) #'Apodization: 0=uniform, 1=hamming'
   
   # position profile
   capon.profilePos = [-13.83174473246271, 96.50653026882506, 0.0]
   
   ax = []
   for i in range(4):
      figure = pl.figure()
      ax.append(figure.add_subplot(1,1,1))     
   
   Nbs = [0, 3]
   
   for Nb in Nbs:
   
      capon.Nb.updateValue(Nb)
      if capon.Nb.value > 0:
         capon.d.updateValue(d_BS)
      else:
         capon.d.updateValue(d_ES)
   
      capon.processData()
      
      capon.plot(ax[0],ax[1],ax[2], ax[3])
      
      if Nb == 0:
         x = capon.x_aslice
         das_profile = capon.img_das_aslice
         capon_profile = capon.img_cap_aslice
      else:
         capon_bs_profile = capon.img_cap_aslice
         
   fig = pl.figure()
   ax = fig.add_subplot(1,1,1)
            
   ax.plot(x, das_profile, c=(0.25,0.25,1.00), ls=':', lw=2.0, label='DAS')
   ax.plot(x, capon_profile, c=(1.00,0.25,0.25), ls='-.', lw=2.0, label='ES-Capon')
   ax.plot(x, capon_bs_profile, c=(0.0,0.0,0.0), ls='-', lw=2.0, label='BS-Capon')
   
   ax.set_ylim([capon.minDynRange.value, capon.maxDynRange.value])
   
   ax.set_title('Lateral intensity at %d mm range'%capon.profilePos[1], fontsize='large')
   ax.set_xlabel('Width [mm]', fontsize='large')
   ax.set_ylabel('Lateral intensity [dB]', fontsize='large')
   if legendOn:
      ax.legend(loc=1, markerscale=0.5)
   
   #fig.savefig('invivo_slice', bbox_inches='tight', pad_inches=0.5)
   fig.savefig('invivo_slice.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
   
def findResolution(caponProcessor, srcFrames, resLimit, p_depth, fps=25.0, highres=True):
   
   capon = caponProcessor
   frames = srcFrames
   
   for frame in frames:
      
      capon.image_idx.updateValue(frame)
      capon.processData()
      capon.interpolateData()
      
      p1 =  0.5 + (1+1.5)*frame/fps 
      p2 = -0.5 + (1-1.5)*frame/fps
      pc = (p1 + p2) / 2
      p = [np.array([p1, p_depth]), np.array([p2, p_depth]), np.array([pc, p_depth])]
      amp_p = []
      
      for i in range(len(p)):
         p_dist = np.linalg.norm(p[i])
         p_angle = np.arctan(p[i][0]/p[i][1])
         idx_range = np.argmin(abs(1000*capon.ranges_intrp-p_dist))
         idx_angle = np.argmin(abs(capon.angles_intrp-p_angle))
         
         if highres: 
            amp_p.append(capon.img_cap_detected[idx_range][idx_angle])                      
         else:
            amp_p.append(capon.img_das_detected[idx_range][idx_angle])
      
      res = (amp_p[0] + amp_p[1])/2 - amp_p[2]
       
      if (res > resLimit):
         print 'Resolution measured to ', p1-p2,  ' from', amp_p, ' in frame ', frame
         return p1 - p2
   
def runTradoffAnalysis():
   datatag = 'motion phantom 16x 3.4 MHz 20dB30dB'
   #filename = 'motion_phantom_16x'
   #datatag.append('Vingmed data liver 2STB 1')
   #datatag.append('Vingmed data liver 2STB 2')
   #datatag.append('Vingmed data cardiac 2STB 1')
   #datatag.append('Vingmed data cardiac 4MLA 1')
      
   multiFile = True
    
   capon = CaponProcessor(datatag, multiFile)
   
   # set parameters for capon processor
   capon.image_idx.updateValue(0)
   
   capon.sliceRangeStart.updateValue(120)#(100)
   capon.sliceAngleStart.updateValue(195)#(350)#200
   capon.sliceRangeEnd.updateValue(145)#(120)#0
   capon.sliceAngleEnd.updateValue(780)#(600)
   
   capon.Ky.updateValue(4) # incoherent interpolation in range
   capon.Kx.updateValue(4) # in azimuth
   
   capon.Nb.updateValue(0)
   
   #capon.L.updateValue()
   capon.K.updateValue(2)
   
   capon.minDynRange.updateValue(-20)
   capon.maxDynRange.updateValue(35)
   capon.minDynRangeCapon.updateValue(-20)
   capon.maxDynRangeCapon.updateValue(35)
   
   #capon.apod.updateValue(0) #'Apodization: 0=uniform, 1=hamming'
   
   # position profile
   #capon.profilePos = [0.0, 93.0, 0.0]
   
   #Nb_L_range = range(32, 2, -1)
   Nb_L_range = range(31, 0, -2)
   
   fps = 25.0
   p_depth = 90.0#93.0 #mm
   #p_depth += 0.2148 # add 1/4 of the pulse length
   p_depth += 0.1388#compensate for extra travel distance in elevation
   
   frames = range(7, 30)
   
   res_limit = 6.0
   
   fig = pl.figure()
   ax = fig.add_subplot(1,1,1)
   
   for b in [True, False]:
      
      resolution = []
      
      if not b:
         capon.L.resetValue()
         capon.d.updateValue(-1)#1)
      else:
         capon.d.updateValue(-1)#1)
   
      for v in Nb_L_range:
         
         print 'Value is: ', v, ' b is:', b
         
         if b:
            capon.L.updateValue(v)
         else:
            capon.Nb.updateValue(v)
            #capon.d.updateValue((1.0/30.0)*v - 61.0/30.0) # make d go from 0.1 at Nb=31 to 0.01 at Nb=1
            
         resolution.append(findResolution(capon, frames, res_limit, p_depth, fps, highres=True))
               
      if b:
         ax.plot(Nb_L_range, resolution, c=(0.0,0.0,0.0), ls='-', marker='o', lw=2.0, label='Reducing subarrays')
      else:
         ax.plot(Nb_L_range, resolution, c=(0.2,0.2,0.2), ls='-', marker='s', lw=2.0, label='Reducing beamspace')
         
   das_res = findResolution(capon, frames, res_limit, p_depth, fps, highres=False)
   ax.plot(Nb_L_range, das_res*np.ones(len(Nb_L_range)), c=(0.4,0.4,0.4), ls='--', lw=2.0, label='Delay-and-sum')
   
   #ax.set_ylim(ax.get_ylim()[0]-0.1, ax.get_ylim()[1]+0.1)
   ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1]+0.1)      
   ax.set_xlim([Nb_L_range[0], Nb_L_range[-1]])
   ax.set_title('Trading resolution for speed', fontsize='x-large')
   ax.set_xlabel('$L$ or $N_b$', fontsize='large')
   ax.set_ylabel('Resolution [mm]', fontsize='large')      
   ax.legend(loc=2, markerscale=0.5)
   fig.savefig('speed_res_trade.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
   
   
if __name__ == '__main__':
   #process_and_plot_phantom_image()
   #process_and_plot_invivo_image()
   #process_and_plot_phantom_image(3)
   #process_and_plot_invivo_image(3)
   
   plot_slices_simulation()
   #plot_slices_invivo(False)
   
   #runTradoffAnalysis()
