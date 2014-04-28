
#include <cuda_runtime.h>
//#include <helper_cuda.h>

#define cutilSafeCall(err)           __cudaSafeCall      (err, __FILE__, __LINE__)

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
   if( cudaSuccess != err) {
      //fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error %d: %s.\n",
      //         file, line, (int)err, cudaGetErrorString( err ) );
      //exit(-1);
   }
}


int main()
{
   cutilSafeCall( cudaDeviceReset() );
//   cudaDeviceReset();
   return 0;
}
