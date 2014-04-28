# Find Python modules.
#
# This code sets the following variables:
#
# NUMPY_INCLUDE_DIR
#
# NOTE!
# This file is VERY short of being a full-blown find-all-python-modules scipt.
# I will simply add to this whenever I need to.

include( FindPackageHandleStandardArgs )

execute_process ( COMMAND python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()" OUTPUT_VARIABLE SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE )

find_path( NUMPY_INCLUDE_DIR numpy/arrayobject.h
           ${SITE_PACKAGES}/numpy/core/include )
           #${SITE_PACKAGES}/numpy/core/include )
#message(${SITE_PACKAGES}/numpy/core/include/numpy/arrayobject.h)
         
# Attempt to search the ~/.local folder as well  
if( NOT ${NUMPY_INCLUDE_DIR} )
    execute_process ( COMMAND find $ENV{HOME}/.local -iname "site-packages" OUTPUT_VARIABLE SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE )
    find_path( NUMPY_INCLUDE_DIR numpy/arrayobject.h
               ${SITE_PACKAGES}/numpy/core/include )
endif( )

# If that still did not work, try to probe a path relative to the python include path
if( NOT ${NUMPY_INCLUDE_DIR} )
    set( SITE_PACKAGES ${PYTHON_INCLUDE_DIR}/../Lib/site-packages )
    find_path( NUMPY_INCLUDE_DIR numpy/arrayobject.h
               ${SITE_PACKAGES}/numpy/core/include )
endif()

find_package_handle_standard_args( Numpy DEFAULT_MSG
                                   NUMPY_INCLUDE_DIR )
