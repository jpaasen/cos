
cython_add_module( mynumpy_tools mynumpy_tools.pyx )


# link_directories( ${CMAKE_CURRENT_SOURCE_DIR}/linalg )

# Process the CMakeLists.txt in the subdirectories
add_subdirectory( linalg )
add_subdirectory( beamformer )

add_subdirectory( lib )