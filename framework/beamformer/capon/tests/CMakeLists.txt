

#include_directories( .. )

cython_add_module( testRTC testRTC.pyx )

target_link_libraries( testRTC RealTimeCapon )

