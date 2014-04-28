# Note: when executed in the build dir, then CMAKE_CURRENT_SOURCE_DIR is the
# build dir.
# file( COPY setup.py src test bin DESTINATION "${CMAKE_ARGV3}"
# FILES_MATCHING PATTERN "*.py" )

# message(STATUS "${CMAKE_ARGV3}")
# message(STATUS "${CMAKE_CURRENT_BINARY_DIR}")

# file( GLOB_RECURSE pattern_files RELATIVE
#    "${CMAKE_CURRENT_SOURCE_DIR}/" "*.py" )
# foreach( pattern_file ${pattern_files} )
# message(STATUS ${pattern_file})
# add_custom_command(
#    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${pattern_file}"
#    COMMAND cmake -E copy
#    "${CMAKE_CURRENT_SOURCE_DIR}/${pattern_file}"
#    "${CMAKE_CURRENT_BINARY_DIR}/${pattern_file}"
#    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${pattern_file}"
# )
# list( APPEND pattern_files_dest "${pattern_file}" )
# 
# endforeach( pattern_file )
# add_custom_target( CopyPatterns ALL DEPENDS ${pattern_files_dest} ) 


# file( COPY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION "${CMAKE_ARGV3}" 
# FILES_MATCHING PATTERN "*.py" EXCLUDE build )

include( FindPackageHandleStandardArgs )

cmake_policy(SET CMP0009 NEW) # Tells CMake that it shouldn't follow symlinks

list(APPEND searchPaths "cuda" "cmake" "framework" "apps"  )

if(CMAKE_BUILD_TYPE STREQUAL Release)

   foreach(searchPath ${searchPaths})
      file(GLOB_RECURSE pyFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${searchPath}/*.py)
      file(GLOB_RECURSE mFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${searchPath}/*.m)
      file(GLOB_RECURSE txtFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${searchPath}/*.txt)
   
       #file(GLOB_RECURSE pyFiles RELATIVE ${searchPath} *.py)
       #file(GLOB_RECURSE mFiles RELATIVE ${searchPath} *.m)
       #file(GLOB_RECURSE txtFiles RELATIVE ${searchPath} README.txt INSTALL.txt)
       list(APPEND templateFiles ${templateFiles} ${pyFiles} ${mFiles} ${txtFiles})
   endforeach()
   
   file(GLOB rootFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py INSTALL.txt)
   list(APPEND templateFiles ${templateFiles} ${rootFiles})
   
else()

   foreach(searchPath ${searchPaths})
       # message(STATUS "Configuring directory ${CMAKE_ARGV3}")
       file(MAKE_DIRECTORY ${CMAKE_ARGV3})
       #file(GLOB_RECURSE allFiles RELATIVE ${searchPath} *.py *.pyx *.npy *.m *.cpp *.h *.txt *.cmake)
       file(GLOB_RECURSE pyFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${searchPath}/*.py)
       file(GLOB_RECURSE pyxFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${searchPath}/*.pyx)
       file(GLOB_RECURSE npyFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${searchPath}/*.npy)
       file(GLOB_RECURSE mFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${searchPath}/*.m)
       file(GLOB_RECURSE cppFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${searchPath}/*.cpp)
       file(GLOB_RECURSE hFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${searchPath}/*.h)
       file(GLOB_RECURSE cuFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${searchPath}/*.cu)
       file(GLOB_RECURSE cuhFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${searchPath}/*.cuh)
       file(GLOB_RECURSE txtFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${searchPath}/*.txt)
       file(GLOB_RECURSE cmakeFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${searchPath}/*.cmake)
       file(GLOB_RECURSE matplotlibFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${searchPath}/*matplotlibrc)
       
       list(APPEND templateFiles ${templateFiles} ${pyFiles} ${pyxFiles} ${npyFiles} ${mFiles} ${cppFiles} ${hFiles} ${cuFiles} ${cuhFiles} ${txtFiles} ${cmakeFiles} ${matplotlibFiles})
       
       #foreach(templateFile ${templateFiles})
       #   list(APPEND absoluteTemplateFiles ${absoluteTemplateFiles} ${searchPath}/templateFile)
          #set(templateFile ${searchPath}/templateFile)
       #endforeach()
       #${searchPath}
       #list(APPEND templateFiles ${templateFiles} ${pyFiles} ${npyFiles} ${mFiles} ${cppFiles} ${hFiles} ${txtFiles} ${cmakeFiles})
   endforeach()
   
   file(GLOB rootFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.py "cmake_clean" *.txt *.cmake)
   list(APPEND templateFiles ${templateFiles} ${rootFiles})
   
endif()


foreach(templateFile ${templateFiles})
   get_filename_component(file_path ${templateFile} PATH)
   if(NOT EXISTS ${CMAKE_ARGV3}/${file_path})
      file(MAKE_DIRECTORY ${CMAKE_ARGV3}/${file_path})
   endif()
endforeach()
# file(GLOB_RECURSE cuFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cu)
# file(GLOB_RECURSE hFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.h)
# file(GLOB_RECURSE cuhFiles RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cuh)
# list(APPEND templateFiles ${pyFiles} ${pyxFiles} ${npyFiles} ${mFiles} ${cppFiles} ${cuFiles} ${hFiles} ${cuhFiles})

if( UNIX )
   foreach(templateFile ${templateFiles})
   set(srcTemplatePath ${CMAKE_CURRENT_SOURCE_DIR}/${templateFile})
   if(NOT IS_DIRECTORY ${srcTemplatePath})
      if(NOT EXISTS ${CMAKE_ARGV3}/${templateFile})
         #message(${CMAKE_ARGV3}/${templateFile})
         execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${srcTemplatePath} ${CMAKE_ARGV3}/${templateFile} )
      endif()
   endif()
   endforeach(templateFile)    
else()
   foreach(templateFile ${templateFiles})
   set(srcTemplatePath ${CMAKE_CURRENT_SOURCE_DIR}/${templateFile})
   if(NOT IS_DIRECTORY ${srcTemplatePath})
      configure_file(
         ${srcTemplatePath}
         ${CMAKE_ARGV3}/${templateFile}
         COPYONLY)
   endif()
   endforeach(templateFile)  
endif()
#       execute_process(
#          COMMAND ${CMAKE_COMMAND}
#          -E copy_if_different ${srcTemplatePath} ${CMAKE_ARGV3}/${templateFile}
#          )


# macro(COPY_IF_DIFFERENT source dest
# 
# endmacro(MACRO(COPY_IF_DIFFERENT)
