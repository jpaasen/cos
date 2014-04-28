 
if( USE_MKL )

   # Split the fused getCaponAllC into a pyx and py file:
   cython_add_module( mklUnivariateSplineC mklUnivariateSplineC.pyx mklUnivariateSpline.cpp )
    
   #target_link_libraries( mkl_UnivariateSplineC libmkl_intel_lp64.a )
    target_link_libraries( mklUnivariateSplineC mkl_rt )
    
   # Split the fused getCaponAllC into a pyx and py file:
   cython_add_module( mklcSolveCholeskyC mklcSolveCholeskyC.pyx mklcSolveCholesky.cpp )


   #target_link_libraries( mkl_UnivariateSplineC libmkl_intel_lp64.a )

   target_link_libraries( mklcSolveCholeskyC mkl_rt )
endif()

