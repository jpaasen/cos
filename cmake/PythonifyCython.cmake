

function( cython_add_fused_module _module_name fused_files )
#   get_cmake_property(_variableNames VARIABLES)
#   foreach (_variableName ${_variableNames})
#   message(STATUS "${_variableName}=${${_variableName}}")
#   endforeach()

   # message(" Current source directory ${CMAKE_CURRENT_SOURCE_DIR}")

   add_custom_target( "${_module_name} (defuse)" ALL ${CMAKE_COMMAND} -P
      ${CMAKE_SOURCE_DIR}/cmake/PythonifyCythonFunction.cmake ${_module_name} "${CMAKE_CURRENT_SOURCE_DIR}" "${fused_files}" ) 
      
   # Compile the pyx-file
   cython_add_module( ${_module_name} ${fused_files} )
#   file( REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/${pyxfile} ))
   
endfunction()