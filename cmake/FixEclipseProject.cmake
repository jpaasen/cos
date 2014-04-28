
# if( "${CMAKE_GENERATOR}" MATCHES "Unix Makefiles" )
if( EXISTS ${CMAKE_BINARY_DIR}/.project )

   file( READ ${CMAKE_BINARY_DIR}/.project ECLIPSE_PROJECT )
   #file( READ ${CMAKE_BINARY_DIR}/cmake/FixEclipseProject.cmake.filters ECLIPSE_FILTERS )

#    message( ${ECLIPSE_FILTERS} )

   string( REPLACE "@" "-"  MODIFIED_ECLIPSE_PROJECT ${ECLIPSE_PROJECT} )
   #string( REPLACE "</linkedResources>\n</projectDescription>" "${ECLIPSE_FILTERS}"  MODIFIED_ECLIPSE_PROJECT ${MODIFIED_ECLIPSE_PROJECT} )

   file( WRITE ${CMAKE_BINARY_DIR}/.project ${MODIFIED_ECLIPSE_PROJECT} )

endif()