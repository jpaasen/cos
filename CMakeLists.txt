#The name of the project
#project(Framework)

# if( UNIX )
#    set( CMAKE_C_COMPILER "icc" )
#    set( CMAKE_CXX_COMPILER "icpc" )
# endif()

#The CMake Minimum version that is required. The FindCUDA script
#is distributed since version 2.8, and we aim to use it.
cmake_minimum_required(VERSION 2.8)

include(CheckTypeSize)
check_type_size("void*" SIZEOF_VOID_P BUILTIN_TYPES_ONLY)

message( ${SIZEOF_VOID_P} )


# Make the scripts available in the 'cmake' directory available for the
# 'include()' command, 'find_package()' command.
set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_CURRENT_SOURCE_DIR})


# Include configuration file, i.e. settings that differ from computer to computer
if( EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake )
   include( Config )
else()
   include( ConfigExample )
endif()

# To suport spesification of build type from the command line  
if(DEFINED BT)
   set( CMAKE_BUILD_TYPE ${BT} )
endif()
   
# Make sure one of the valid build types are selected:
if(CMAKE_BUILD_TYPE STREQUAL Release)
   message("Build type: Release")
elseif(CMAKE_BUILD_TYPE STREQUAL Debug)
   message("Build type: Debug")
   add_definitions(-DVERBOSE)
elseif(CMAKE_BUILD_TYPE STREQUAL Profile)
   message("Build type: Profile")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS_RELEASE}")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_DEBUG} ${CMAKE_C_FLAGS_RELEASE}")
   add_definitions(-UVERBOSE)
else()
   message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}")
endif()


# ??? What have you done here Jo? use 64bit?
#set_property(GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS 1)


# Add sweet colors to make output process (optional, of course :)
# set( CMAKE_VERBOSE_MAKEFILE on ) 
set(CMAKE_COLOR_MAKEFILE ON)
  

# Include all directories that contain a CMakeLists.txt by default
set(CMAKE_INCLUDE_CURRENT_DIR ON) 


########
# UNIX #
########

if (UNIX)
   add_definitions(-Wall -Wextra)
endif(UNIX) 


#################
# VISUAL STUDIO #
#################

# Is now taken care by install command!
#if (MSVC90) # If Visual Studio... 
   #... move shared lib out of Release/Debug folders
   #   message(STATUS "You are using Visual Studio 2008")
   #set(CMAKE_STATIC_LIBRARY_PREFIX "../")
   #set(CMAKE_SHARED_LIBRARY_PREFIX "../")
   #set(PYTHON_MODULE_PREFIX "../")
   #endif()


########################
# SYSTEM INCLUDE PATHS #
########################

#find_path(STDIO_INCLUDE_PATH stdio.h) # only linux?
#include_directories("${STDIO_INCLUDE_PATH}/dummy/../")


###################
# PYTHON / CYTHON #
###################

# Python is used throughout, might aswell include the headers here
find_package(PythonModules)
include_directories( ${NUMPY_INCLUDE_DIR} )

# The Eclipse generator creates a project name with an '@' in it. Pydev dislikes this,
# so we call this little cmake script to make that '@' into an '-' instead.
add_custom_target( FixEclipseProject ALL ${CMAKE_COMMAND} -P
   ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FixEclipseProject.cmake
   ${CMAKE_CURRENT_BINARY_DIR} )

# With CMake, a clean separation can be made between the source tree and the
# build tree. When all source is compiled, as with pure C/C++, the source is
# no-longer needed in the build tree. However, with pure *.py source, the
# source is processed directly. To handle this, we reproduce the availability
# of the source files in the build tree.
# add_custom_target( FixEclipseProject ALL sed s+'@'+'-'+ ${CMAKE_BINARY_DIR}/.project
# WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} VERBATIM)

# Include the CMake script UseCython.cmake. This defines add_cython_module().
# Instruction for use can be found at the top of cmake/UseCython.cmake.
include(UseCython)
include(PythonifyCython)
include(PythonifyCythonFunction)

# Set some Cython options
set(CYTHON_ANNOTATE ON)
set(CYTHON_FLAGS "--verbose" "-X profile=False,cdivision=True,wraparound=False,boundscheck=False")
# set(CYTHON_FLAGS "--verbose" "-X profile=False,cdivision=False,wraparound=True,boundscheck=True")
#set(CYTHON_FLAGS "-X profile=False,cdivision=True,wraparound=False,boundscheck=False")

# Unset the Py_DEBUG flag in python source code to avoid VS linking against the python
# debug library (which doesn't seem to exist by default on windows machines)
if( WIN32 )
  add_definitions(-UPy_DEBUG)
endif()

#########################
# REPLICATE SOURCE TREE #
#########################

add_custom_target( ReplicateSourceTree ALL ${CMAKE_COMMAND} -P
   ${CMAKE_SOURCE_DIR}/cmake/ReplicateSourceTree.cmake
   ${CMAKE_BINARY_DIR}
   WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} )


########
# CUDA #
########

find_package(CUDA)
include(FindCUDA)

# If ${CUDA_SDK_ROOT_DIR} is empty, make sure NVSDKCUDA_ROOT environment variable is set
include_directories(${CUDA_INCLUDE_DIRS})

#message(${CUDA_SDK_ROOT_DIR})
include_directories("${CUDA_SDK_ROOT_DIR}/common/inc")

# Since some CUDA versions are short of some header files we need, we've copied these
# to this folder:
include_directories(${CMAKE_SOURCE_DIR}/cuda/include)


#message(${CUDA_SDK_ROOT_DIR})
#message(${CUDA_SDK_ROOT_DIR}/common/lib/linux/x86_64)
# Put SDK libs in library search path
if ( UNIX )
   #link_directories("${CUDA_SDK_ROOT_DIR}/common/lib/linux/x86_64")
   get_filename_component( CUDA_LIBS ${CUDA_CUBLAS_LIBRARIES} PATH ) 
   link_directories( ${CUDA_LIBS} )
   #message( ${CUDA_SDK_ROOT_DIR}/lib )
elseif ( WIN32 )
   if (CMAKE_CL_64)
      link_directories( ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64 )
      link_directories( ${CUDA_SDK_ROOT_DIR}/common/lib/x64 )
   else ()
      link_directories( ${CUDA_TOOLKIT_ROOT_DIR}/lib/Win32 )
      link_directories( ${CUDA_SDK_ROOT_DIR}/common/lib/Win32 )
   endif ()
endif ()

#message( "   Build Type: ${CMAKE_BUILD_TYPE}" )
# Setup build flags for Windows and Linux 
if(CMAKE_BUILD_TYPE STREQUAL Debug)
   if( WIN32 )
      set( NVCC_FLAGS -g --ptxas-options=-v)# -G0) # -g is for host debug info, -G0 is for device debug info
   else()
      set( NVCC_FLAGS -g -G -lineinfo --ptxas-options=-v --profile -Xcompiler -fPIC )
   endif()
endif()
if(CMAKE_BUILD_TYPE STREQUAL Release)
   if( WIN32 )
      set( NVCC_FLAGS -use_fast_math -O --ptxas-options=-v )
   else()
      set( NVCC_FLAGS -use_fast_math -O --ptxas-options=-v -Xcompiler -fPIC )
#      set( NVCC_FLAGS -O --ptxas-options=-v -Xcompiler -fPIC )
   endif()
endif()
if(CMAKE_BUILD_TYPE STREQUAL Profile)
   if( WIN32 )
      #set( NVCC_FLAGS -g --ptxas-options=-v)# -G0) # -g is for host debug info, -G0 is for device debug info
      set( NVCC_FLAGS -g -use_fast_math -O --ptxas-options=-v )
   else()
      #set( NVCC_FLAGS -g -G -lineinfo --ptxas-options=-v --profile -Xcompiler -fPIC )
      set( NVCC_FLAGS -g -G -use_fast_math -O --ptxas-options=-v -Xcompiler -fPIC )
   endif()
endif()
#
set(CUDA_NVCC_FLAGS ${NVCC_FLAGS}
   #-gencode arch=compute_11,code=\\\"compute_11,sm_11\\\"
   #-gencode arch=compute_13,code=\\\"compute_13,sm_13\\\"
   -gencode arch=compute_20,code=\\\"compute_20,sm_20\\\"
   -gencode arch=compute_20,code=\\\"compute_20,sm_21\\\"
)

#######
# MEX #
#######

find_package(Matlab-rwp)
include(MatlabMakeMacros)

# Compile with the '-g' flag? 
if(CMAKE_BUILD_TYPE STREQUAL Debug)
   set(MEX_DEBUG_SYMBOLS ON)
endif()
if(CMAKE_BUILD_TYPE STREQUAL Profile)
   set(MEX_DEBUG_SYMBOLS ON)
endif()

# dotMatlab.cpp does not work when compiled in 64bit mode, so we'll compile
# with 32bit compatibility enabled
#set( MEX_32BIT_COMPATIBLE ON )s


# MKL
#######

if(USE_MKL)
   find_package(MKL)
   include_directories( ${MKL_INCLUDE_DIR} )
   link_directories( ${INTEL_LIBRARY_DIR} ${MKL_LIBRARY_DIR} )
   add_definitions(-DUSE_MKL)
endif()



############
# INCLUDES #
############

include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/framework/lib )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/framework/linalg )
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/framework/beamformer/capon/RealTimeCapon )
 

##################
# SUBDIRECTORIES #
##################

add_subdirectory( framework )
#add_subdirectory( sandbox )

########################
# PRINT ALL VARIABLES? #
########################

if( PRINT_VARIABLES )
   get_cmake_property(_variableNames VARIABLES)
   foreach (_variableName ${_variableNames})
   message(STATUS "${_variableName}=${${_variableName}}")
   endforeach()
endif()


###########
# Apps    #
###########

add_subdirectory( apps )
