
include_directories( ${CMAKE_CURRENT_SOURCE_DIR} )

#cuda_add_executable( MatrixMul  matrixMul.cu matrixMul_kernel.cuh matrixMul_gold.cpp)

if ( UNIX )   
   #target_link_libraries( MatrixMul cutil_x86_64 )
else ()
   # cutil is built as a dynamic lib on windows
   #target_link_libraries( MatrixMul cutil64 )
endif ()   
   
# Make sure 'position independent code' is created, to enable
# this static library to be embedded in a shared one later
#if( UNIX )
#   set_target_properties( RealTimeCapon PROPERTIES COMPILE_FLAGS -fPIC)
#endif( UNIX )
