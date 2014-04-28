
# Building the Library
if(MSVC_IDE) # If Visual Studio...
 
   #... group all headers into a vs project
   file(GLOB_RECURSE INCS "*.h")
   file(GLOB_RECURSE CUDA_INCS "*.cuh")
   source_group(headers FILES ${INCS} ${CUDA_INCS})

   file(GLOB_RECURSE SRC "*.cpp")
   file(GLOB_RECURSE CUDA_SRC "*.cu")
   source_group(source FILES ${SRC} ${CUDA_SRC})
   
   #set_source_files_properties(sliding_beamspace_kernel.cu PROPERTIES CUDA_NVCC_FLAGS "-G0")
   #SET_SOURCE_FILES_PROPERTIES( ${LIST_OF_FILES} PROPERTIES COMPILE_FLAGS "-O3") 

   cuda_add_library( RealTimeCapon STATIC Capon.cpp BuildR.cpp CudaUtils.cpp capon_kernel.cu buildR_kernel.cu nvidia_solve.cu solver3x3_kernel.cu solver1x1_kernel.cu Solver.cpp sliding_beamspace_kernel.cu ${INCS} ${CUDA_INCS})

   # Testing cuda_compile_ptx
   #cuda_compile_ptx(ptx_files sliding_beamspace_kernel.cu)
   #cuda_add_library(ptx STATIC ${ptx_files})

   # Other way of compiling CUDA code. 
   # cuda_compile(file_list, file0, file1 ... [STATIC | SHARED | MODULE] [OPTIONS ...])
   # cuda_add_library(lib_name, file_list) or add_library(lib_name, file_list)

else()

   include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

   cuda_add_library( RealTimeCapon STATIC Capon.cpp BuildR.cpp CudaUtils.cpp capon_kernel.cu buildR_kernel.cu nvidia_solve.cu solver3x3_kernel.cu solver1x1_kernel.cu Solver.cpp sliding_beamspace_kernel.cu )
   
endif()
   
target_link_libraries( RealTimeCapon PrintIt ${CUDA_CUBLAS_LIBRARIES})

#set( NVCC_FLAGS -use_fast_math ${NVCC_FLAGS} )
   
# Make sure 'position independent code' is created, to enable
# this static library to be embedded in a shared one later
if( UNIX )
   set_target_properties( RealTimeCapon PROPERTIES COMPILE_FLAGS -fPIC)
endif( UNIX )

cython_add_module( Solver Solver.pyx )
target_link_libraries( Solver RealTimeCapon )

if( UNIX )
	add_subdirectory( profile )
endif()