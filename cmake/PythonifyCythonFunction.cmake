
set( _module_name ${CMAKE_ARGV3} )
set( source_directory ${CMAKE_ARGV4} )
set( fused_files ${CMAKE_ARGV5} )

#message( "_module_name ${CMAKE_ARGV3}" )
#message( "source_directory ${CMAKE_ARGV4}" )
#message( "fused_files ${CMAKE_ARGV5}" )
#message( "CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}" )
#message( "CMAKE_CURRENT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}" )

#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()


foreach( fused_file ${fused_files} )
   
  if( EXISTS ${source_directory}/${fused_file} )

     file( READ ${source_directory}/${fused_file} FUSED_FILE_CONTENTS )
     
     # Generate Python file by removing Cython contents:
     string( REGEX REPLACE "###CYTHON_MODE[^§]*§"  ""  PY_FILE_CONTENTS  ${FUSED_FILE_CONTENTS} )
     string( REGEX REPLACE "\"\"\"PYTHON_MODE" "" PY_FILE_CONTENTS  ${PY_FILE_CONTENTS} )
     string( REGEX REPLACE "PYTHON_MODE_END\"\"\""  ""  PY_FILE_CONTENTS  ${PY_FILE_CONTENTS} )
     
     # Generate Cython file by removing Python contents:
#     string( REGEX REPLACE "###CYTHON_MODE"  ""  PYX_FILE_CONTENTS  ${FUSED_FILE_CONTENTS} )
#     string( REGEX REPLACE "\"\"\"PYTHON_MODE.*PYTHON_MODE_END\"\"\""  ""  PYX_FILE_CONTENTS  ${PYX_FILE_CONTENTS} )
     
     # Generate filenames for the two files
#     string( REPLACE ".fused.pyx" ".pyx"  pyxfile ${fused_file} )
     string( REPLACE ".pyx" "Py.py"  pyfile ${fused_file} )
    
     # Store the files
#     file( WRITE ${CMAKE_CURRENT_BINARY_DIR}/${pyxfile} ${PYX_FILE_CONTENTS} )
     file( WRITE ${CMAKE_CURRENT_BINARY_DIR}/${pyfile} ${PY_FILE_CONTENTS} )

  else()
     message(FATAL "${source_directory}/${fused_file} not found")
  
  endif()
      
endforeach()