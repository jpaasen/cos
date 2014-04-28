Files currently included in compilation of the library (VS2008 on Windows, Make on Linux):

- IBuildR.h
- BuildR (.cu/.h)
- buildR_kernel.cu
- ICapon.h
- Capon (.cpp/.h)
- cudaConfig.h
- CudaUtils (.cpp/.h)
- Isolver.h
- SolveNvidia (.cpp/.h)

(Nvidia solver)
- inverse (.cu/.h)
- solve (.cu/.h)
- operations.h

All .cu files must be compiled with nvcc (except for buildR_kernel which is directly included into BuildR.cu)


Additional include directories:
$(CUDA_PATH)\include

Additional depending libs and directories:
cadart.lib (.so on linux?)
$(CUDA_PATH)\lib\($(PlatformName)?)
