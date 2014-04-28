
import sys, re
from subprocess import Popen, PIPE
import framework.mynumpy as np

info = {}

QUIET=True
ONLY_TIME=True
USE_CODE_COUNTERS=True

import framework.beamformer.capon.getCaponCUDA as getCaponCUDA
#sys.path.append('/home/me/Work/UiO/Phd/Code/Profile')

def getTimings(app_name='testEfficiency_default',
                 functions=['gauss_solve','buildR_kernel','amplitude_capon'],
                 M=8,L=4):
   
   if USE_CODE_COUNTERS:
      prof = Popen("./%s testEfficiency testMVDRKernelPerformance %d %d"%(app_name,M,L), shell=True, stdout=PIPE, stderr=PIPE)
      output, error = prof.communicate()
      
      return np.zeros((3,))
   
   def run(M=24,L=8,verbose=False):
   
#      print "nvprof -u us -t 0.1 --devices 0 %s ./%s testEfficiency testMVDRKernelPerformance %d %d"%(events,app_name,M,L
      prof = Popen("nvprof -u ns %s ./%s testEfficiency testMVDRKernelPerformance %d %d"%(events,app_name,M,L), shell=True, stdout=PIPE, stderr=PIPE)
      
      output, error = prof.communicate()
      
      if error == '':
         if verbose:
            print output
         return output
      else:
         print output
         raise Exception(error)
     
   events=''
   profiler_output = run(M,L)
         
   lines = re.split("\n+", profiler_output)
   runtimes = np.zeros([])
   entered_once = False
   for l in lines:
      
      for i,function in enumerate(functions):
         
         if re.search(".+%s.+"%function, l):             
            timings = re.sub("^\s*", '', l)
            timings = re.sub("[\sa-zA-Z]*%s.+"%function, '', timings)         
            columns = re.split("\s+", timings)
            
            if not entered_once:
               entered_once = True
               runtimes = np.zeros((functions.__len__(),columns.__len__()))
               
            runtimes[i] = map(float,columns)
            break
               
         else:
            pass
      
   return runtimes[:,3]/1e3

def getMemoryOps(app_name='testEfficiency_default',
                 functions=['gauss_solve','buildR_kernel','amplitude_capon'],
                 M=16,L=4):
   
   if not ONLY_TIME:
      def run(M=24,L=8,verbose=False):
      
   #      print "nvprof -u us --devices 0 --events %s ./%s testEfficiency testMVDRKernelPerformance %d %d"%(events,app_name,M,L)
         prof = Popen("nvprof -u ns --devices 0 --events %s ./%s testEfficiency testMVDRKernelPerformance %d %d"%(events,app_name,M,L), shell=True, stdout=PIPE, stderr=PIPE)
         
         output, error = prof.communicate()
         
         if error == '':
            if verbose:
               print output
            return output
         else:
            print output
            raise Exception(error)
      
      events = "gld_request,gst_request,shared_load,shared_store"
      events_list = re.split(',',events)
      profiler_output = run(M,L)
            
      lines = re.split("\n+", profiler_output)
      memops = np.zeros((functions.__len__(),events_list.__len__()))
      for i,l in enumerate(lines):
         
         for j,function in enumerate(functions):
            
            if re.search(".+%s.+"%function, l):
               
               for k,event in enumerate(events_list):
                  
                  timings = re.sub("^\s*", '', lines[i+k+1])
                  timings = re.sub("\s*%s.*"%event, '', timings)
                  columns = re.split("\s+", timings)
                  
                  memops[j,k] = float(columns[1])
                     
            else:
               pass
         
   #   print "Recorded runtimes [ms]: "
   #   print runtimes[:,3]/1e3
      
      return memops
   else:
      return np.zeros((3,))
   
def getInstructions(app_name='testEfficiency_default',
                 functions=['gauss_solve','buildR_kernel','amplitude_capon'],
                 M=16,L=8):
   
   if not ONLY_TIME:
      def run(M=16,L=4,verbose=False):
      
   #      print "nvprof -u us --devices 0 --events %s ./%s testEfficiency testMVDRKernelPerformance %d %d"%(events,app_name,M,L)
         prof = Popen("nvprof -u ns --devices 0 --events %s ./%s testEfficiency testMVDRKernelPerformance %d %d"%(events,app_name,M,L), shell=True, stdout=PIPE, stderr=PIPE)
         
         output, error = prof.communicate()
         
         if error == '':
            if verbose:
               print output
            return output
         else:
            print output
            raise Exception(error)
      
      events   = "inst_issued1_0,inst_issued2_0,inst_issued1_1,inst_issued2_1"#,\
   #               l2_read_requests,l2_write_requests,l2_read_texture_requests"
      events_list = re.split(',',events)
      profiler_output = run(M,L)
            
      lines = re.split("\n+", profiler_output)
      instructions = np.zeros((functions.__len__(),events_list.__len__()))
      for i,l in enumerate(lines):
         
         for j,function in enumerate(functions):
            
            if re.search(".+%s.+"%function, l):
               
               for k,event in enumerate(events_list):
                  
                  timings = re.sub("^\s*", '', lines[i+k+1])
                  timings = re.sub("\s*%s.*"%event, '', timings)
                  columns = re.split("\s+", timings)
                  
                  instructions[j,k] = float(columns[1])
                     
            else:
               pass
         
      ##From CUPTI Users manual:
      ## inst_issued1_0 + (inst_issued2_0 * 2) + inst_issued1_1 + (inst_issued2_1 * 2)

   #   print instructions
      instructions = instructions[:,0] + 2*instructions[:,1] + instructions[:,2] + 2*instructions[:,3]

   #   print "Recorded runtimes [ms]: "
   #   print runtimes[:,3]/1e3
      
      return instructions

   ##   events   = "--events inst_issued1_0,inst_issued2_0,inst_issued1_1,inst_issued2_1"#,\
   ##                     l2_read_requests,l2_write_requests,l2_read_texture_requests"
   else:
      return np.zeros((3,))

def collectResults(M=32,L=16):

   if QUIET:
      print "Collecting results for M=%d, L=%d"%(M,L)
      
   if not QUIET:
      print "############"
      print "## DEFAULT #"
      print "############"
      print ""
   
   
   functions=['gauss_solve','buildR_kernel','amplitude_capon']
   
   timings_default      = getTimings(functions=functions,M=M,L=L) 
   memops_default       = getMemoryOps(functions=functions,M=M,L=L)
   instructions_default = getInstructions(functions=functions,M=M,L=L)
   
   if not QUIET:
      print 'Runtimes: %d %d %d'%tuple(timings_default)
      
      print ''
      print 'Memory instructions (gld_request gst_request shared_load shared_store):'
      for i,key in enumerate(functions):
         a = [functions[i]]
         a.extend(memops_default[i])
         print '%s: %d %d %d %d'%tuple(a)
         
      print ''
      print 'Instructions %d %d %d'%tuple(instructions_default)
   
   
   #print "############"
   #print "## DEFAULT #"
   #print "############"
   #print ""
   #print "Runtimes: 40700 2510 742"
   #print ""
   #print "Memory instructions (gld_request gst_request shared_load shared_store):"
   #print "gauss_solve: 160000 10000 7380000 2860000"
   #print "buildR_kernel: 10000 85000 555000 95000"
   #print "amplitude_capon: 15000 10000 245000 20000"
   #print ""
   #print "Instructions 65520464 7268280 2220345"
   
   
   
   if not QUIET:
      print ""
      print "###########"
      print "## MEMORY #"
      print "###########"
      print ""
   
   timings_memory      = getTimings(app_name='testEfficiency_memcheck', functions=functions,M=M,L=L) 
   memops_memory       = getMemoryOps(app_name='testEfficiency_memcheck', functions=functions,M=M,L=L)
   instructions_memory = getInstructions(app_name='testEfficiency_memcheck', functions=functions,M=M,L=L)
   
   if not QUIET:
      print 'Runtimes [ms]: %d %d %d'%tuple(timings_memory)
      
      print ''
      print 'Memory instructions diff (should be all zeros):'
      for i,key in enumerate(functions):
         a = [functions[i]]
         a.extend(memops_memory[i]-memops_default[i])
         print '%s: %d %d %d %d'%tuple(a)
         
      print ''
      print 'Instructions %d %d %d'%tuple(instructions_memory)
   
      
   if not QUIET:
      print ""
      print "################"
      print "## MATH GLOBAL #"
      print "################"
      print ""
   
   timings_math      = getTimings(app_name='testEfficiency_mathcheck_global', functions=functions,M=M,L=L) 
   memops_math       = getMemoryOps(app_name='testEfficiency_mathcheck_global', functions=functions,M=M,L=L)
   instructions_math = getInstructions(app_name='testEfficiency_mathcheck_global', functions=functions,M=M,L=L)
   
   if not QUIET:
      print 'Runtimes [ms]: %d %d %d'%tuple(timings_math)
      
      print ''
      print 'Memory instructions diff (should be all zeros):'
      for i,key in enumerate(functions):
         a = [functions[i]]
         a.extend(memops_math[i]-memops_default[i])
         print '%s: %d %d %d %d'%tuple(a)
         
      print ''
      print 'Instructions %d %d %d'%tuple(instructions_math)
      
      print ""
      print "################"
      print "## MATH SHARED #"
      print "################"
      print ""
   
   timings_math      = getTimings(app_name='testEfficiency_mathcheck_shared', functions=functions,M=M,L=L) 
   memops_math       = getMemoryOps(app_name='testEfficiency_mathcheck_shared', functions=functions,M=M,L=L)
   instructions_math = getInstructions(app_name='testEfficiency_mathcheck_shared', functions=functions,M=M,L=L)
   
   if not QUIET:
      print 'Runtimes [ms]: %d %d %d'%tuple(timings_math)
      
      print ''
      print 'Memory instructions diff (should be all zeros):'
      for i,key in enumerate(functions):
         a = [functions[i]]
         a.extend(memops_math[i]-memops_default[i])
         print '%s: %d %d %d %d'%tuple(a)
         
      print ''
      print 'Instructions %d %d %d'%tuple(instructions_math)
   
   if not QUIET:
      print ""
      print "#########"
      print "## MATH #"
      print "#########"
      print ""
   
   timings_math      = getTimings(app_name='testEfficiency_mathcheck', functions=functions,M=M,L=L) 
   memops_math       = getMemoryOps(app_name='testEfficiency_mathcheck', functions=functions,M=M,L=L)
   instructions_math = getInstructions(app_name='testEfficiency_mathcheck', functions=functions,M=M,L=L)
   
   if not QUIET:
      print 'Runtimes [ms]: %d %d %d'%tuple(timings_math)
      
      print ''
      print 'Memory instructions diff (should be all zeros):'
      for i,key in enumerate(functions):
         a = [functions[i]]
         a.extend(memops_math[i]-memops_default[i])
         print '%s: %d %d %d %d'%tuple(a)
         
      print ''
      print 'Instructions %d %d %d'%tuple(instructions_math)
   
   if not QUIET:   
      print ""
      print "#########"
      print "## TOTAL #"
      print "#########"
      print ""
      
#   time_total  = timings_default[0] + timings_default[1]
#   time_math   = timings_math[0] + timings_math[1]
#   time_memory = timings_memory[0] + timings_memory[1]
   
   return np.array([timings_default[0],timings_default[1],
                    timings_math[0],   timings_math[1],
                    timings_memory[0], timings_memory[1]])
      
   ####################
   ### USING PROFILER #
   ####################
   #
   ## Analysing instruction / byte ratio with the use of the profiler:
   #
   #
   ##   events   = "--events inst_issued1_0,inst_issued2_0,inst_issued1_1,inst_issued2_1"#,\
   ##                     l2_read_requests,l2_write_requests,l2_read_texture_requests"
   ##   default_event_run = run()
   #
   ##From CUPTI Users manual:
   ## inst_issued1_0 + (inst_issued2_0 * 2) + inst_issued1_1 + (inst_issued2_1 * 2)
   #
   #
   #
   #
   ##events   = "--events gld_request,gst_request,shared_load,shared_store"
   ##
   ##
   ##lines = re.split("\n+", default_run)
   ##k = -1
   #
   #
   ## Inst_per_SM = 32 * instructions_issued (32 - warp size)
   ## Mem_per_SM = 32B * (l2_read_requests+l2_write_requests+l2_read_texture_requests)
   #
   ## Compare Inst_per_SM / Mem_per_SM
   #
   #
   #
   #
   ##for l in lines:
   ##   if re.search("\s+Kernel.+", l):
   ##      kernels.append(table)
   ##            
   ##   else:
   ##      table.append( re.split("\s+",l) )
   ##
   ###########
   ## MEMORY #
   ###########
   #
   #app_name = "testEfficiency_memcheck"
   #memory_run = run()
   #
   #print "Check that removing math did not reduce load/store instruction count... "
   #events   = "--events gld_request,gst_request,shared_load,shared_store"
   #memory_event_run = run(verbose=True)
   ##
   ##
   ##########
   ### MATH #
   ##########
   ##
   ##app_name = "testEfficiency_mathcheck"
   ##math_run = run()
   ##events   = "--events gld_request,gst_request,shared_load,shared_store"
   ##math_event_run = run()
   #
   #
   #return info
   ##export PYTHONPATH="/home/me/Work/UiO/Phd/Code/Profile"
   ##python testEfficiency.py testMVDRKernelPerformance
   #
   ##nvprof ./testEfficiency testEfficiency testMVDRKernelPerformance

M = 8
L_list = np.arange(M-1)+2
time = np.zeros((L_list.shape[0],6))
for l,L in enumerate(L_list):
   time[l] = collectResults(M,L)
np.savetxt('time-M%d.txt'%M,time)

M = 16
L_list = np.arange(M-1)+2
time = np.zeros((L_list.shape[0],6))
for l,L in enumerate(L_list):
   time[l] = collectResults(M,L)
np.savetxt('time-M%d.txt'%M,time)

M = 32
L_list = np.arange(M-1)+2
time = np.zeros((L_list.shape[0],6))
for l,L in enumerate(L_list):
   time[l] = collectResults(M,L)
np.savetxt('time-M%d.txt'%M,time)
         

