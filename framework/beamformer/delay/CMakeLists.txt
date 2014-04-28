 

# Split the fused getCaponAllC into a pyx and py file:
cython_add_fused_module( tdibC tdibC.pyx )

#target_link_libraries( getCaponC RealTimeCapon )

if( UNIX )

   # Split the fused getCaponAllC into a pyx and py file:
   cython_add_fused_module( tdibMKLC tdibMKLC.pyx )
   
endif()

add_subdirectory( gpu )