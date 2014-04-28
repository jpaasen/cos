# Find Python modules.
#
# This code sets the following variables:
#
# NUMPY_INCLUDE_DIR
#
# NOTE!
# This file is VERY short of being a full-blown find-all-python-modules scipt.
# I will simply add to this whenever I need to.

include( FindPackageHandleStandardArgs )

if(UNIX)
  set( HARDPATHS "/opt/intel" )
  set( LIB_SUFFIX "so" )
  set( LIB_PREFIX "lib" )
else() #if(WIN32)
  set( HARDPATHS "C:/Program Files (x86)/Intel/Composer XE 2011 SP1/mkl" )
  set( LIB_SUFFIX "lib" )
  set( LIB_PREFIX "" )
endif()

#file(GLOB MKL_ROOT_DIR "${HARDPATHS}" "$ENV{MKL_ROOT}")
file(GLOB INTEL_ROOT_DIR "${HARDPATHS}" "$ENV{MKL_ROOT}")

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
 set(ARCH_SUFFIX ia32)
else()
 set(ARCH_SUFFIX intel64)
endif()

find_path( MKL_INCLUDE_DIR mkl.h
        ${INTEL_ROOT_DIR}/mkl/include )
        
find_path( INTEL_LIBRARY_DIR "${LIB_PREFIX}imf.${LIB_SUFFIX}"
        ${INTEL_ROOT_DIR}/lib/${ARCH_SUFFIX} )

find_path( MKL_LIBRARY_DIR "${LIB_PREFIX}mkl_core.${LIB_SUFFIX}"
        ${INTEL_ROOT_DIR}/mkl/lib/${ARCH_SUFFIX} )

find_package_handle_standard_args( INTEL DEFAULT_MSG
                                INTEL_LIBRARY_DIR )

find_package_handle_standard_args( MKL DEFAULT_MSG
                                MKL_INCLUDE_DIR )
                                
find_package_handle_standard_args( MKL DEFAULT_MSG
                                MKL_LIBRARY_DIR )
                                   
#endif()
