
cuda_add_executable(cudaDeviceReset cudaDeviceReset.cu)
#target_link_libraries( cudaDeviceReset ${CUDA_LIBRARIES} )
#install( TARGETS cudaDeviceReset DESTINATION ${CMAKE_BINARY_DIR}/bin )
 
#include_directories( ${CMAKE_SOURCE_DIR}/framework/beamformer/capon/gpu/LibRealTimeCapon )
get_directory_property( DirDefs DIRECTORY ${CMAKE_SOURCE_DIR} COMPILE_DEFINITIONS )
cython_add_module( TestCaseBasec TestCaseBasec.pyx )

add_library( PrintIt STATIC printit.cpp )

if( UNIX )
set_target_properties( PrintIt PROPERTIES COMPILE_FLAGS -fPIC)
endif( UNIX )

add_subdirectory( mkl )
