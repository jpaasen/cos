#include_directories( .. )

cuda_add_executable( testEfficiency_default testEfficiency.cpp testEfficiency.py ../Capon.cpp ../BuildR.cpp ../CudaUtils.cpp ../capon_kernel.cu buildR_kernel.cu nvidia_solve.cu ../solver3x3_kernel.cu ../Solver.cpp ../sliding_beamspace_kernel.cu ../solver3x3_kernel.cu ../solver1x1_kernel.cu
                     OPTIONS )
target_link_libraries( testEfficiency_default ${PYTHON_LIBRARIES} PrintIt ${CUDA_CUBLAS_LIBRARIES} )


cuda_add_executable( testEfficiency_memcheck testEfficiency.cpp testEfficiency.py ../Capon.cpp ../BuildR.cpp ../CudaUtils.cpp ../capon_kernel.cu buildR_kernel.cu nvidia_solve.cu ../solver3x3_kernel.cu ../Solver.cpp ../sliding_beamspace_kernel.cu ../solver3x3_kernel.cu ../solver1x1_kernel.cu
                     OPTIONS -DMEMORY_ONLY -DPROFILE )
target_link_libraries( testEfficiency_memcheck ${PYTHON_LIBRARIES} PrintIt ${CUDA_CUBLAS_LIBRARIES} )

cuda_add_executable( testEfficiency_mathcheck_global testEfficiency.cpp testEfficiency.py ../Capon.cpp ../BuildR.cpp ../CudaUtils.cpp ../capon_kernel.cu buildR_kernel.cu nvidia_solve.cu ../solver3x3_kernel.cu ../Solver.cpp ../sliding_beamspace_kernel.cu ../solver3x3_kernel.cu ../solver1x1_kernel.cu
                     OPTIONS -DMATH_ONLY_GLOBAL -DPROFILE )
target_link_libraries( testEfficiency_mathcheck_global ${PYTHON_LIBRARIES} PrintIt ${CUDA_CUBLAS_LIBRARIES} )

cuda_add_executable( testEfficiency_mathcheck_shared testEfficiency.cpp testEfficiency.py ../Capon.cpp ../BuildR.cpp ../CudaUtils.cpp ../capon_kernel.cu buildR_kernel.cu nvidia_solve.cu ../solver3x3_kernel.cu ../Solver.cpp ../sliding_beamspace_kernel.cu ../solver3x3_kernel.cu ../solver1x1_kernel.cu
                     OPTIONS -DMATH_ONLY_SHARED -DPROFILE )
target_link_libraries( testEfficiency_mathcheck_shared ${PYTHON_LIBRARIES} PrintIt ${CUDA_CUBLAS_LIBRARIES} )

cuda_add_executable( testEfficiency_mathcheck testEfficiency.cpp testEfficiency.py ../Capon.cpp ../BuildR.cpp ../CudaUtils.cpp ../capon_kernel.cu buildR_kernel.cu nvidia_solve.cu ../solver3x3_kernel.cu ../Solver.cpp ../sliding_beamspace_kernel.cu ../solver3x3_kernel.cu ../solver1x1_kernel.cu
                     OPTIONS -DMATH_ONLY_SHARED -DMATH_ONLY_GLOBAL -DPROFILE )
target_link_libraries( testEfficiency_mathcheck ${PYTHON_LIBRARIES} PrintIt ${CUDA_CUBLAS_LIBRARIES} )
