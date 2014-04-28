import os
import cPickle
from scipy import io
import matplotlib

class Colormaps(object):
   pass

#def setColormap():
   
def setColormap():
   
   path = os.getenv('PHDCODE_ROOT')+ '/data/sonar/colormaps'
   c = Colormaps()

   bronze       = io.loadmat(path+'/bronze_color.mat')['color']
   sas          = io.loadmat(path+'/sas_color.mat')['color']
   yellow       = io.loadmat(path+'/yellow_color.mat')['color']
   std_gamma_II = io.loadmat(path+'/std_gamma_II.mat')['color']
   
   c.bronze       = matplotlib.colors.ListedColormap(bronze,       N=bronze.shape[0])
   c.sas          = matplotlib.colors.ListedColormap(sas,          N=bronze.shape[0])
   c.yellow       = matplotlib.colors.ListedColormap(yellow,       N=bronze.shape[0])
   c.std_gamma_II = matplotlib.colors.ListedColormap(std_gamma_II, N=bronze.shape[0])
   
   output = open( path+'/colormaps.pickle','w' )
                 
   cPickle.dump(c, output, cPickle.HIGHEST_PROTOCOL)
   
   output.close()
   
                 
def getColormap():

   input = open(
   os.getenv('PHDCODE_ROOT')+'/data/sonar/colormaps/colormaps.pickle','r' )
   colormaps = cPickle.load(input)
   input.close()
   return colormaps