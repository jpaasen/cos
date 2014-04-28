import cython

cdef extern from "/usr/include/google/profiler.h":
   void ProfilerStart( char* fname )
   void ProfilerStop()
 
def startProfiler(fname):
   ProfilerStart(<char *>fname)
 
def stopProfiler():
   ProfilerStop()