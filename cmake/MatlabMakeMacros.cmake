# This file was found in the source code of Dru F., Fillard P.,
# Vercauteren T., "An ITK Implementation of the Symmetric Log-Domain
# Diffeomorphic Demons Algorithm", Insight Journal, 2009 Jan-Jun
# http://hdl.handle.net/10380/3060

# Configuration options.
set( MEX_DEBUG_SYMBOLS OFF
   CACHE BOOL "Embed debug symbols in the binaries (-g flag)?" )
set( MEX_32BIT_COMPATIBLE ON
   CACHE BOOL "Should mex file be compiled in 32bit compatible mode?" )

mark_as_advanced( CYTHON_ANNOTATE CYTHON_NO_DOCSTRINGS CYTHON_FLAGS )

#message( ${SIZEOF_VOID_P} )   

macro(LOAD_REQUIRED_PACKAGE Package)
  LOADPACKAGE(${Package})
  IF(NOT ${Package}_FOUND)
    message(FATAL_ERROR "Required package ${Package} was not found.\n
    Look at Find${Package}.cmake in the CMake module directory for clues
    on what you're supposed to do to help find this package.  Good luck.\n")
  endif(NOT ${Package}_FOUND)
endmacro(LOAD_REQUIRED_PACKAGE)

macro(LOAD_OPTIONAL_PACKAGE Package)
  LOADPACKAGE(${Package} QUIET)
endmacro(LOAD_OPTIONAL_PACKAGE)

macro(ADD_MEX_FILE Target)
  INCLUDE_DIRECTORIES(${MATLAB_INCLUDE_DIR})
  ADD_LIBRARY(${Target} SHARED ${ARGN})
  target_link_libraries(${Target} 
    ${MATLAB_MX_LIBRARY} 
    ${MATLAB_MEX_LIBRARY} 
    ${MATLAB_MAT_LIBRARY}
    )
    
   if( MEX_DEBUG_SYMBOLS )
      set( DEBUG_SYMBOLS "-g" )
   else( MEX_DEBUG_SYMBOLS )
      set( DEBUG_SYMBOLS "" )
   endif( MEX_DEBUG_SYMBOLS )
     
   if( MEX_32BIT_COMPATIBLE )
      set( 32BIT_COMPATIBLE "-DMX_COMPAT_32" )
   else( MEX_32BIT_COMPATIBLE )
      set( 32BIT_COMPATIBLE "" )
   endif( MEX_32BIT_COMPATIBLE )
     
  set_target_properties(${Target} PROPERTIES PREFIX "")
  
  include(CheckTypeSize)
  check_type_size("void*" SIZEOF_VOID_P BUILTIN_TYPES_ONLY)
  
  # Determine mex suffix
  if(UNIX)
    if(APPLE)
      if(CMAKE_OSX_ARCHITECTURES MATCHES i386)
        set_target_properties(${Target} PROPERTIES SUFFIX ".mexmaci")
      else(CMAKE_OSX_ARCHITECTURES MATCHES i386)
        set_target_properties(${Target} PROPERTIES SUFFIX ".mexmac")
      endif(CMAKE_OSX_ARCHITECTURES MATCHES i386)
    else(APPLE)
      if(SIZEOF_VOID_P MATCHES "4")
        set_target_properties(${Target} PROPERTIES SUFFIX ".mexglx")
      elseif(SIZEOF_VOID_P MATCHES "8")
        set_target_properties(${Target} PROPERTIES SUFFIX ".mexa64")
      else(SIZEOF_VOID_P MATCHES "4")
      message( SIZEOF_VOID_P )
        message(FATAL_ERROR 
          "SIZEOF_VOID_P (${SIZEOF_VOID_P}) doesn't indicate a valid platform")
      endif(SIZEOF_VOID_P MATCHES "4")
    endif(APPLE)
  elseif(WIN32)
    if(SIZEOF_VOID_P MATCHES "4")
      set_target_properties(${Target} PROPERTIES SUFFIX ".mexw32")
    elseif(SIZEOF_VOID_P MATCHES "8")
      set_target_properties(${Target} PROPERTIES SUFFIX ".mexw64")
    else(SIZEOF_VOID_P MATCHES "4")
      message(FATAL_ERROR 
        "SIZEOF_VOID_P (${SIZEOF_VOID_P}) doesn't indicate a valid platform")
    endif(SIZEOF_VOID_P MATCHES "4")
  endif(UNIX)
  
  if(MSVC)
    set(matlab_flags "-DMATLAB_MEX_FILE" "${DEBUG_SYMBOLS}" "${32BIT_COMPATIBLE}")
    sd_append_target_properties(${Target} COMPILE_FLAGS ${MATLAB_FLAGS})
    set_target_properties(${Target} PROPERTIES LINK_FLAGS "/export:mexFunction")
  else(MSVC)
    if(SIZEOF_VOID_P MATCHES "4")
      set(MATLAB_FLAGS "-fPIC" "-D_GNU_SOURCE" "-pthread"
      "-D_FILE_OFFSET_BITS=64" "-DMX_COMPAT_32" "${DEBUG_SYMBOLS}")
    else(SIZEOF_VOID_P MATCHES "4")
      set(MATLAB_FLAGS "-fPIC" "-D_GNU_SOURCE" "-pthread"
      "-D_FILE_OFFSET_BITS=64" "${DEBUG_SYMBOLS}" "${32BIT_COMPATIBLE}")
    endif(SIZEOF_VOID_P MATCHES "4")
    sd_append_target_properties(${Target} COMPILE_FLAGS ${MATLAB_FLAGS})
    
    if(APPLE)
      if(CMAKE_OSX_ARCHITECTURES MATCHES i386)
        # mac intel
        set_target_properties(${Target} PROPERTIES 
          LINK_FLAGS "-L${MATLAB_SYS} -Wl,-flat_namespace -undefined suppress")
      else(CMAKE_OSX_ARCHITECTURES MATCHES i386)
        # mac powerpc?
        set_target_properties(${Target} PROPERTIES 
          LINK_FLAGS "-L${MATLAB_SYS} -Wl,-flat_namespace -undefined suppress")
      endif(CMAKE_OSX_ARCHITECTURES MATCHES i386)
    eLSE(APPLE)
      IF(SIZEOF_VOID_P MATCHES "4")
        set_target_properties(${Target} PROPERTIES 
          LINK_FLAGS "-Wl,-E -Wl,--no-undefined")
      elseif(SIZEOF_VOID_P MATCHES "8")
        set_target_properties(${Target} PROPERTIES 
          LINK_FLAGS "-Wl,-E -Wl,--no-undefined")
      else(SIZEOF_VOID_P MATCHES "4")
        message(FATAL_ERROR 
          "SIZEOF_VOID_P (${SIZEOF_VOID_P}) doesn't indicate a valid platform")
      endif(SIZEOF_VOID_P MATCHES "4")
    endif(APPLE)
  endif(MSVC)
endmacro(ADD_MEX_FILE)


macro(SD_APPEND_TARGET_PROPERTIES TARGET_TO_CHANGE PROP_TO_CHANGE)
  foreach(_newProp ${ARGN})
    get_target_property(_oldProps ${TARGET_TO_CHANGE} ${PROP_TO_CHANGE})
    iF(_oldProps)
      IF(NOT "${_oldProps}" MATCHES "^.*${_newProp}.*$")
        set_target_properties(${TARGET_TO_CHANGE} PROPERTIES ${PROP_TO_CHANGE} "${_newProp} ${_oldProps}")
      endif(NOT "${_oldProps}" MATCHES "^.*${_newProp}.*$")
    eLSE(_oldProps)
      set_target_properties(${TARGET_TO_CHANGE} PROPERTIES ${PROP_TO_CHANGE} ${_newProp})
    endif(_oldProps)
  endforeach(_newProp ${ARGN})
endmacro(SD_APPEND_TARGET_PROPERTIES TARGET_TO_CHANGE PROP_TO_CHANGE)


macro(sd_add_link_libraries Target)
  foreach (currentLib ${ARGN})
    if (${currentLib}_LIBRARIES)
      target_link_libraries(${Target} ${${currentLib}_LIBRARIES})
    elseif (${currentLib}_LIBRARY)
      target_link_libraries(${Target} ${${currentLib}_LIBRARY})
    else (${currentLib}_LIBRARIES)
      #message("WARNING: ${currentLib}_LIBRARY and ${currentLib}_LIBRARIES are undefined. Using ${currentLib} in linker")
      target_link_libraries(${Target} ${currentLib})
    endif (${currentLib}_LIBRARIES)
    
    if (${currentLib}_INCLUDE_DIRS)
      INCLUDE_DIRECTORIES(${${currentLib}_INCLUDE_DIRS})
    elseif (${currentLib}_INCLUDE_DIR)
      INCLUDE_DIRECTORIES(${${currentLib}_INCLUDE_DIR})
    else (${currentLib}_INCLUDE_DIRS)
      #message("WARNING: ${currentLib}_INCLUDE_DIR and ${currentLib}_INCLUDE_DIR are undefined. No specific include dir will be used for ${currentLib}")
    endif (${currentLib}_INCLUDE_DIRS)
  endforeach (currentLib)
endmacro(sd_add_link_libraries)


macro(SD_UNIT_TEST Src)
  # remove extension
  string(REGEX REPLACE "[.][^.]*$" "" Target ${Src})

  # parse arguments
  set(currentPos "")
  set(testLibs "")
  set(testExtLibs "")
  set(testArgs "")

  foreach (arg ${ARGN})
    if (arg STREQUAL "LIBS")
      set(currentPos "LIBS")
    elseif (arg STREQUAL "EXTLIBS")
      set(currentPos "EXTLIBS")
    elseif (arg STREQUAL "ARGS")
      set(currentPos "ARGS")
    else (arg STREQUAL "LIBS")
      if (currentPos STREQUAL "LIBS")
        set(testLibs ${testLibs} ${arg})
      elseif (currentPos STREQUAL "EXTLIBS")
        set(testExtLibs ${testExtLibs} ${arg})
      elseif (currentPos STREQUAL "ARGS")
        set(testArgs ${testArgs} ${arg})
      else (currentPos STREQUAL "ARGS")
         message(FATAL_ERROR "Unknown argument")
      endif (currentPos STREQUAL "LIBS")
    endif (arg STREQUAL "LIBS")
  endforeach (arg ${ARGN})

  # setup target
  add_executable(${Target} ${Src})
  sd_add_link_libraries(${Target} ${testExtLibs})
  target_link_libraries(${Target} ${testLibs})
  set_target_properties(${Target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/tests ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/tests)
  add_test(${Target} ${PROJECT_BINARY_DIR}/tests/${Target} ${testArgs})
endmacro(SD_UNIT_TEST)


macro(SD_EXECUTABLE Src)
  # remove extension
  string(REGEX REPLACE "[.][^.]*$" "" Target ${Src})

  # parse arguments
  set(currentPos "")
  set(appLibs "")
  set(appExtLibs "")

  foreach (arg ${ARGN})
    if (arg STREQUAL "LIBS")
      set(currentPos "LIBS")
    elseif (arg STREQUAL "EXTLIBS")
      set(currentPos "EXTLIBS")
    else (arg STREQUAL "LIBS")
      if (currentPos STREQUAL "LIBS")
        set(appLibs ${appLibs} ${arg})
      elseif (currentPos STREQUAL "EXTLIBS")
        set(appExtLibs ${appExtLibs} ${arg})
      else (currentPos STREQUAL "LIBS")
         message(FATAL_ERROR "Unknown argument")
      endif (currentPos STREQUAL "LIBS")
    endif (arg STREQUAL "LIBS")
  endforeach (arg ${ARGN})

  # setup target
  add_executable(${Target} ${Src})
  sd_add_link_libraries(${Target} ${appExtLibs})
  target_link_libraries(${Target} ${appLibs})
  set_target_properties(${Target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
endmacro(SD_EXECUTABLE)
