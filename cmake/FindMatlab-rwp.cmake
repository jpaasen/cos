# - This module looks for Matlab and associated development libraries
# Defines:
#  MATLAB_INCLUDE_DIR: include path for mex.h, engine.h
#  MATLAB_LIBRARIES:        required libraries: libmex, etc
#  MATLAB_MEX_LIBRARY:      path to libmex.lib
#  MATLAB_MX_LIBRARY:       path to libmx.lib
#  MATLAB_ENG_LIBRARY:      path to libeng.lib
#  MATLAB_MEX_VERSION_FILE: path to mexversion.rc or mexversion.c
#  MATLAB_MEX_SUFFIX:       filename suffix for mex-files (e.g. '.mexglx' or '.mexw64')

# This version modified by RW Penney, November 2008
# $Revision: 27 $, $Date: 2008-12-22 11:47:45 +0000 (Mon, 22 Dec 2008) $

include( FindPackageHandleStandardArgs )

set(MATLAB_FOUND 0)

if(WIN32)

  file(GLOB _auto_matlab_prefixes "C:/Program Files/MATLAB/R20*" "C:/Program Files/MATLAB/R2011a" "C:/matlab" "$ENV{MATLAB_ROOT}")

  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    # Regular x86
    set(MATLAB_MEX_SUFFIX mexw32)
    set(_extern_arch "win32")
  else(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(MATLAB_MEX_SUFFIX mexw64)
    set(_extern_arch "win64")
  endif(CMAKE_SIZEOF_VOID_P EQUAL 4)

  set( _matlab_path_suffixes
       "extern/lib/${_extern_arch}/microsoft"
       "extern/lib/${_extern_arch}/microsoft/msvc71"  # VS 8,9,10
       "extern/lib/${_extern_arch}/microsoft/msvc70"  # VS 7.0 
       "extern/lib/${_extern_arch}/microsoft/msvc60"  # VS 6
       "extern/lib/win32/microsoft/bcc54"             # Borland
      )
       
  # Search for available compilers:
  # (This would be neater using 'ELSEIF', but that isn't available until cmake-2.4.4)
#  if( IS_DIRECTORY 
#  if(${CMAKE_GENERATOR} MATCHES "Visual Studio 6")
#    set(_matlab_path_suffixes "extern/lib/${_extern_arch}/microsoft/msvc60"
#        "extern/lib/${_extern_arch}/microsoft")
#  endif(${CMAKE_GENERATOR} MATCHES "Visual Studio 6")
#  if(${CMAKE_GENERATOR} MATCHES "Visual Studio 7")
#    set(_matlab_path_suffixes "extern/lib/${_extern_arch}/microsoft/msvc70"
#        "extern/lib/${_extern_arch}/microsoft")
#  endif(${CMAKE_GENERATOR} MATCHES "Visual Studio 7")
#  if(${CMAKE_GENERATOR} MATCHES "Visual Studio [891]*")
#    set(_matlab_path_suffixes "extern/lib/${_extern_arch}/microsoft/msvc71"
#        "extern/lib/${_extern_arch}/microsoft")
#  endif(${CMAKE_GENERATOR} MATCHES "Visual Studio [891]*")
#  if(${CMAKE_GENERATOR} MATCHES "Borland")
#    set(_matlab_path_suffixes "extern/lib/win32/microsoft/bcc54")
#  endif(${CMAKE_GENERATOR} MATCHES "Borland")
#  if(NOT _matlab_path_suffixes)
#    message(FATAL_ERROR "Generator not compatible: ${CMAKE_GENERATOR}")
#  endif(NOT _matlab_path_suffixes)

  set(_libmex_name "libmex")
  set(_libmx_name "libmx")
  set(_libeng_name "libeng")

else(WIN32)

  file(GLOB _auto_matlab_prefixes "$ENV{MATLAB_ROOT}" "/usr/local/matlab-*" "/opt/matlab-*" "/opt/matlab/*")

  if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    # Regular x86
    set(_matlab_path_suffixes "bin/glnx86")
    set(MATLAB_MEX_SUFFIX mexglx)
  else(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(_matlab_path_suffixes "bin/glnxa64")
    set(MATLAB_MEX_SUFFIX mexa64)
  endif(CMAKE_SIZEOF_VOID_P EQUAL 4)

  set(_libmex_name "mex")
  set(_libmx_name "mx")
  set(_libeng_name "eng")

endif(WIN32)


set(_matlab_path_prefixes
  ${MATLAB_PATH_PREFIXES}
  ${_auto_matlab_prefixes}
  ${MATLAB_ROOT}
)

# Search for include-files & libraries using architecture-dependent paths:
message("${MATLAB_INCLUDE_DIR}")
foreach(_matlab_prefix ${_matlab_path_prefixes})
  message("searching ${_matlab_prefix}")
  #if(NOT MATLAB_INCLUDE_DIR)
    message("Tying to find ${_matlab_prefix}/extern/include")
    find_path(MATLAB_INCLUDE_DIR "mex.h"
      ${_matlab_prefix}/extern/include)

    if(MATLAB_INCLUDE_DIR)
      set(MATLAB_ROOT ${_matlab_prefix}
        CACHE PATH "Matlab installation directory")
      if(WIN32)
        SET(MATLAB_MEX_VERSIONFILE "${_matlab_prefix}/extern/include/mexversion.rc")
      else(WIN32)
        set(MATLAB_MEX_VERSIONFILE "${_matlab_prefix}/extern/src/mexversion.c")
      endif(WIN32)
    endif(MATLAB_INCLUDE_DIR)
  #endif(NOT MATLAB_INCLUDE_DIR)

  foreach(_matlab_path_suffix ${_matlab_path_suffixes})
    set(_matlab_libdir ${_matlab_prefix}/${_matlab_path_suffix})
#     message("Searching ${_matlab_prefix} ... ${_matlab_libdir}")
    if(NOT MATLAB_MEX_LIBRARY)
      find_library(MATLAB_MEX_LIBRARY ${_libmex_name} ${_matlab_libdir})
      find_library(MATLAB_MX_LIBRARY ${_libmx_name} ${_matlab_libdir})
      find_library(MATLAB_ENG_LIBRARY ${_libeng_name} ${_matlab_libdir})
    endif(NOT MATLAB_MEX_LIBRARY)
  endforeach(_matlab_path_suffix)
endforeach(_matlab_prefix)

set(MATLAB_LIBRARIES
  ${MATLAB_MEX_LIBRARY}
  ${MATLAB_MX_LIBRARY}
  ${MATLAB_ENG_LIBRARY}
)

find_package_handle_standard_args( Matlab DEFAULT_MSG
                                          MATLAB_INCLUDE_DIR
                                          MATLAB_LIBRARIES )

if(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)
  set(MATLAB_FOUND 1)
endif(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)

mark_as_advanced(
  MATLAB_LIBRARIES
  MATLAB_MEX_LIBRARY
  MATLAB_MX_LIBRARY
  MATLAB_ENG_LIBRARY
  MATLAB_INCLUDE_DIR
  MATLAB_MEX_SUFFIX
  MATLAB_MEX_VERSIONFILE
  MATLAB_FOUND
)

# vim: set ts=2 sw=2 et:
