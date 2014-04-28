

# Not sure if these are needed? 
option(SEPARATE_BUILD_TREE "Build " ON)
option(ECLIPSE_CDT4_GENERATE_SOURCE_PROJECT TRUE)

set( USE_MKL FALSE )

# Select build
#set( CMAKE_BUILD_TYPE Debug )
set( CMAKE_BUILD_TYPE Release )
message( "Set build type: ${BT}" )

# Uncomment to pass "#define verbose" to all source files 
#add_definitions(-DVERBOSE)

# Use system programs, or bundled ones? This is for project packaging.
set( USE_BUNDLED_PROGRAMS FALSE )

# Bundling is only supported for 64 bit Linux (yet):
if( UNIX )
   if(CMAKE_SIZEOF_VOID_P EQUAL 4) #32 bit
      set( USE_BUNDLED_PROGRAMS FALSE )
   else(CMAKE_SIZEOF_VOID_P EQUAL 4) #64 bit
      message( "${CMAKE_CURRENT_SOURCE_DIR}/sys/linux64" )
      file(GLOB SYS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/sys/linux64")
   endif(CMAKE_SIZEOF_VOID_P EQUAL 4)
else(UNIX)
   set( USE_BUNDLED_PROGRAMS FALSE )
endif(UNIX)
