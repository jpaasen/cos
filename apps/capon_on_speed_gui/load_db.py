import sys, os
COS_ROOT = os.environ['COS_ROOT']

try:    os.remove("%s/data/configurations.pck"%COS_ROOT)
except: pass
try:    os.remove("%s/data/database.hdf"%COS_ROOT)
except: pass

from framework.Config import Config

c = Config()

def load_db():
   nMotionPhantomFrames = 50
   
   for i in range(1,nMotionPhantomFrames+1):
      s = c.loadFile('motion phantom 16x 3.4 MHz 20dB30dB', i)
      s.save('motion phantom 16x 3.4 MHz 20dB30dB %d'%i)
      del s  
   
load_db()
