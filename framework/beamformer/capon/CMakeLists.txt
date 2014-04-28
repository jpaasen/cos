 
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/RealTimeCapon )

# Split the fused getCaponAllC into a pyx and py file:
cython_add_fused_module( getCaponAllC getCaponAllC.pyx )

if(UNIX)
   cython_add_module( getCaponC getCaponC.pyx )
   #target_link_libraries( getCaponC ${PYTHON_LIBRARIES} )
else()
   cython_add_module( getCaponC getCaponC.pyx )
endif()

# make python interface to CUDA version of Capon
cython_add_module( getCaponCUDA getCaponCUDA.pyx )
target_link_libraries( getCaponCUDA RealTimeCapon )

# make Matlab interface to getCapon (cython and CUDA version)
add_mex_file(getCaponMex2C getCaponMex2C.cpp )
target_link_libraries( getCaponMex2C ${PYTHON_LIBRARIES} getCaponC )

add_mex_file(getCaponMex2CUDA getCaponMex2CUDA.cpp )
target_link_libraries( getCaponMex2CUDA RealTimeCapon )
 
 # Process the CMakeLists.txt in subdirectories.
add_subdirectory( RealTimeCapon )
add_subdirectory( tests )

#add_subdirectory( bundle )

install(TARGETS getCaponAllC getCaponC getCaponCUDA getCaponMex2C DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

