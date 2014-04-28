

if( BUILD_BUNDLED_PROGRAMS )

#    project( GetCaponMATLAB )

    # use, i.e. don't skip the full RPATH for the build tree
#    SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
    
    # when building, don't use the install RPATH already
    # (but later on when installing)
#    SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
    
#    SET(CMAKE_INSTALL_RPATH ${CMAKE_CURRENT_BINARY_DIR}/lib)
    
    # add the automatically determined parts of the RPATH
    # which point to directories outside the build tree to the install RPATH
#¤    SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)
    
    
    # the RPATH to be used when installing, but only if it's not a system directory
#    LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
#    IF("${isSystemDir}" STREQUAL "-1")
#       SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#    ENDIF("${isSystemDir}" STREQUAL "-1")


   message( "${SYS_DIR}" )
   message( "${CMAKE_CURRENT_SOURCE_DIR}" )
   #set( STATIC_PYTHON_LIBRARY "${SYS_DIR}/lib/python2.7/libpython2.7.a" )
   #message( ${PYTHON_LIBRARIES} )
   
   file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/src)
   file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
   file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)
   
   set( HEADER_FILES
      ${CMAKE_SOURCE_DIR}/framework/lib/Complex.h
      ${CMAKE_SOURCE_DIR}/framework/lib/MexArray.h
      ${CMAKE_SOURCE_DIR}/framework/lib/printit.h
      ${CMAKE_SOURCE_DIR}/framework/beamformer/capon/RealTimeCapon/Capon.h
      ${CMAKE_SOURCE_DIR}/framework/beamformer/capon/RealTimeCapon/ICapon.h
      )
      
   get_filename_component( CUDA_BLAS_PATH ${CUDA_CUBLAS_LIBRARIES} PATH )
   #message( "${CUDA_BLAS_PATH}" )
   file(GLOB_RECURSE cudaFiles ${CUDA_BLAS_PATH}/*.so*)
   #message( "${cudaFiles}" )
   
   if( UNIX )
       set( LIBRARY_FILES
       #   ${CMAKE_BINARY_DIR}/framework/beamformer/capon/RealTimeCapon/libRealTimeCapon.a
          #${cudaFiles}
          /opt/cuda/lib64/libcudart.so.5.0
          /opt/cuda/lib64/libcudart.so.5.0.35
          /lib64/libdl.so.2
          /lib64/ld-2.15.so
          /lib64/ld-linux-x86-64.so.2
          /lib32/ld-linux.so.2
          /lib64/libpthread.so.0
          /lib64/librt.so.1
          /usr/lib64/libcuda.so.1
          /lib64/libz.so.1
          /usr/lib/gcc/x86_64-pc-linux-gnu/4.6.2/libstdc++.so.6
          /usr/lib/gcc/x86_64-pc-linux-gnu/4.6.2/libstdc++.so.6.0.16
          /lib64/libm.so.6
          /usr/lib/gcc/x86_64-pc-linux-gnu/4.6.2/libgcc_s.so.1
          /lib64/libc.so.6
          /lib64/libc-2.15.so
          /lib64/libdl-2.15.so
          /lib64/libm-2.15.so
          /lib64/libpthread-2.15.so
          /lib64/librt.so.1
          /lib64/librt-2.15.so
          /lib64/libz.so.1.2.5.1
          /usr/lib64/libcuda.so.304.54
          )
   endif()
      
   
   
      
   set( SOURCE_FILES
      ${CMAKE_SOURCE_DIR}/framework/beamformer/capon/getCaponMex2CUDA.cpp
      ${CMAKE_SOURCE_DIR}/framework/lib/printit.h
      )
      
   install( FILES ${SOURCE_FILES} DESTINATION  ${CMAKE_CURRENT_BINARY_DIR}/src )
   install( FILES ${HEADER_FILES} DESTINATION  ${CMAKE_CURRENT_BINARY_DIR}/include )
      
#   foreach(HEADER_FILE ${HEADER_FILES})
#      get_filename_component( FILE_PATH ${HEADER_FILE} PATH )
#      get_filename_component( FILE_NAME ${HEADER_FILE} NAME )
#      file( COPY ${FILE_PATH}/${FILE_NAME} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/include )
#   endforeach()

   #install( FILES ${LIBRARY_FILES} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/lib )
   #link_directories( ${CMAKE_CURRENT_BINARY_DIR}lib )
   foreach(LIBRARY_FILE ${LIBRARY_FILES})
      get_filename_component( FILE_PATH ${LIBRARY_FILE} PATH )
      get_filename_component( FILE_NAME ${LIBRARY_FILE} NAME )
      
      #add_library( ${FILE_NAME} SHARED IMPORTED )
      #file(GLOB_RECURSE soFiles1 ${FILE_PATH}/${FILE_NAME}.so*)
      file( COPY ${FILE_PATH}/${FILE_NAME} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/lib/ )
      #message( "   Command: ${FILE_PATH}/${FILE_NAME} -> ${CMAKE_CURRENT_BINARY_DIR}/lib/" ) 
   endforeach()
   
#  add_library(pthread SHARED IMPORTED)
#  add_library(cudart SHARED IMPORTED)
#  set_property(TARGET pthread PROPERTY IMPORTED_LOCATION /lib64/libpthread-2.15.so)
#  set_property(TARGET cudart PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/lib/libcudart.so.5.0)
#   foreach(SOURCE_FILE ${SOURCE_FILES})
#      get_filename_component( FILE_PATH ${SOURCE_FILE} PATH )
#      get_filename_component( FILE_NAME ${SOURCE_FILE} NAME )
#      file( COPY ${FILE_PATH}/${FILE_NAME} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/src/ )
#   endforeach()

   

   cuda_add_library( RTCapon STATIC
      ${CMAKE_SOURCE_DIR}/framework/beamformer/capon/RealTimeCapon/Capon.cpp
      ${CMAKE_SOURCE_DIR}/framework/beamformer/capon/RealTimeCapon/BuildR.cpp
      ${CMAKE_SOURCE_DIR}/framework/beamformer/capon/RealTimeCapon/CudaUtils.cpp
      ${CMAKE_SOURCE_DIR}/framework/beamformer/capon/RealTimeCapon/capon_kernel.cu
      ${CMAKE_SOURCE_DIR}/framework/beamformer/capon/RealTimeCapon/buildR_kernel.cu
      ${CMAKE_SOURCE_DIR}/framework/beamformer/capon/RealTimeCapon/nvidia_solve.cu
      ${CMAKE_SOURCE_DIR}/framework/beamformer/capon/RealTimeCapon/solver3x3_kernel.cu
      ${CMAKE_SOURCE_DIR}/framework/beamformer/capon/RealTimeCapon/Solver.cpp
      ${CMAKE_SOURCE_DIR}/framework/beamformer/capon/RealTimeCapon/sliding_beamspace_kernel.cu
      ${CMAKE_SOURCE_DIR}/framework/beamformer/capon/RealTimeCapon/solver3x3_kernel.cu
      ${CMAKE_SOURCE_DIR}/framework/beamformer/capon/RealTimeCapon/solver1x1_kernel.cu
      )
      
   set_target_properties( RTCapon PROPERTIES
#      LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/
#      ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/
      COMPILE_FLAGS -fPIC
      #INSTALL_RPATH  ${CMAKE_CURRENT_BINARY_DIR}/lib
      #INSTALL_RPATH "\$ORIGIN/lib/"
      INSTALL_RPATH lib
      BUILD_WITH_INSTALL_RPATH FALSE
      INSTALL_RPATH_USE_LINK_PATH FALSE
      )
      
   #link_directories( ${CMAKE_CURRENT_BINARY_DIR}/lib )
   target_link_libraries( RTCapon PrintIt cublas
     # cudart pthread
     #${CMAKE_CURRENT_BINARY_DIR}/lib/libcudart.so.5.0
     #${CMAKE_CURRENT_BINARY_DIR}/lib/librt.so.1
   #stdc++ #cuda z dl pthread gcc_s rt
   )
      
   install( TARGETS RTCapon DESTINATION  ${CMAKE_CURRENT_BINARY_DIR}/lib )
   
#   get_cmake_property(_variableNames VARIABLES)
#   foreach (_variableName ${_variableNames})
#   message(STATUS "${_variableName}=${${_variableName}}")
#   endforeach()
   
   #link_directories( lib )
   #SET_TARGET_PROPERTIES(RTCapon PROPERTIES
   # IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/lib/libstdc++.so.6)
    
   #set( CMAKE_SYSTEM_LIBRARY_PATH "" )
   #set( CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "" )
    
#   set( CMAKE_PREFIX_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib )

   


   
   add_mex_file( getCaponGPU ${CMAKE_CURRENT_SOURCE_DIR}/../getCaponMex2CUDA.cpp )
   
   target_link_libraries( getCaponGPU
      RTCapon
   #  ${CMAKE_CURRENT_BINARY_DIR}/lib/libstdc++.so.6
   #  ${CMAKE_CURRENT_BINARY_DIR}/lib/libcuda.so.1
   #  ${CMAKE_CURRENT_BINARY_DIR}/lib/libm.so.6
   #  ${CMAKE_CURRENT_BINARY_DIR}/lib/libdl.so.2
   #  ${CMAKE_CURRENT_BINARY_DIR}/lib/libpthread.so.0
   #  ${CMAKE_CURRENT_BINARY_DIR}/lib/librt.so.1
   )
   

   #message( "Python path" )
   #message( "${SYS_DIR}/lib/python2.7" )
       
   #file(GLOB_RECURSE pySOFiles ${SYS_DIR}/lib/python2.7/*.so)
   #file(GLOB_RECURSE pyAFiles ${SYS_DIR}/lib/python2.7/*.a)
   #list(APPEND pyFiles ${pySOFiles} ${pyAFiles})
   #message( "Listing python files" )
   #message( "${pySOFiles}" )
   
   #file(COPY ${pyFiles} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib/python )
   
   #file(GLOB_RECURSE pyHFiles ${SYS_DIR}/lib/python2.7/*.h)
   #file(COPY ${pyHFiles} DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/include/python )
             
   #foreach(templateFile ${templateFiles})
   #set(srcTemplatePath ${CMAKE_CURRENT_SOURCE_DIR}/${templateFile})
   #   if(NOT IS_DIRECTORY ${srcTemplatePath})
   #      message(STATUS "Copying ${templateFile}")
   #      configure_file(
   #         ${srcTemplatePath}
   #         ${CMAKE_ARGV3}/${templateFile}
   #         COPYONLY)
   #    endif(NOT IS_DIRECTORY ${srcTemplatePath})
   #endforeach(templateFile)
       
    #file( COPY
    #   ${SYS_DIR}/src/Python-2.7.3/Include/*
    #   DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/include/ )
    
    #link_directories( ${CMAKE_CURRENT_SOURCE_DIR}/lib )
    

    #cython_add_module( lib/libgetCaponC ../getCapon.pyx )
    

    # make Matlab interface to getCapon (cython and CUDA version)
    #add_mex_file(getCapon ../getCaponMex.cpp )
    #target_link_libraries( getCapon lib/getCaponC )
     

endif()