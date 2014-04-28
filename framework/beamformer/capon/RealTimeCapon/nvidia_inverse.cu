/*
* Copyright (c) 2011, NVIDIA Corporation. All rights reserved.
*
* Redistribution and use in source and binary forms, with or without 
* modification, are permitted provided that the following conditions are met:
*
*   Redistributions of source code must retain the above copyright notice, 
*   this list of conditions and the following disclaimer.
*
*   Redistributions in binary form must reproduce the above copyright notice,
*   this list of conditions and the following disclaimer in the documentation
*   and/or other materials provided with the distribution.
*
*   Neither the name of NVIDIA Corporation nor the names of its contributors
*   may be used to endorse or promote products derived from this software 
*   without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
* POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include "cuComplex.h"
#include "inverse.h"
#include "operations.h"

#define GRID_DIM_LIMIT  (65520)

#define ARCH_SM13       (0)
#define ARCH_SM20       (1)

#if defined(FERMI)
#define GPU_ARCH        (ARCH_SM20)
#else
#define GPU_ARCH        (ARCH_SM13)
#endif

template <typename T, int arch>
class config {
public:
};

/** ARCH_SM20 configs **/
template<> class config<double,ARCH_SM20> {
public:
   enum { gje3MinDim       =  2 };
   enum { gje3MaxDim       = 77 };
   enum { gje3MinBlks    =    1 };
   enum { gje3MaxThrds   = 1408 }; /* sm_2x, 23 registers per thread */

   enum { gje3DimX_00      = -1 };
   enum { gje3DimX_01      = -1 };
   enum { gje3DimX_02      =  2 };
   enum { gje3DimX_03      =  3 };
   enum { gje3DimX_04      =  4 };
   enum { gje3DimX_05      =  5 };
   enum { gje3DimX_06      =  5 };
   enum { gje3DimX_07      =  7 };
   enum { gje3DimX_08      =  4 };
   enum { gje3DimX_09      =  3 };
   enum { gje3DimX_10      =  6 };
   enum { gje3DimX_11      =  5 };
   enum { gje3DimX_12      =  4 };
   enum { gje3DimX_13      =  4 };
   enum { gje3DimX_14      =  4 };
   enum { gje3DimX_15      =  4 };
   enum { gje3DimX_16      =  4 };
   enum { gje3DimX_17      =  3 };
   enum { gje3DimX_18      =  3 };
   enum { gje3DimX_19      =  3 };
   enum { gje3DimX_20      =  4 };
   enum { gje3DimX_21      =  3 };
   enum { gje3DimX_22      =  4 };
   enum { gje3DimX_23      =  4 };
   enum { gje3DimX_24      =  4 };
   enum { gje3DimX_25      =  5 };
   enum { gje3DimX_26      =  2 };
   enum { gje3DimX_27      =  3 };
   enum { gje3DimX_28      =  4 };
   enum { gje3DimX_29      =  3 };
   enum { gje3DimX_30      =  3 };
   enum { gje3DimX_31      =  3 };
   enum { gje3DimX_32      =  3 };
   enum { gje3DimX_33      =  3 };
   enum { gje3DimX_34      =  4 };
   enum { gje3DimX_35      =  3 };
   enum { gje3DimX_36      =  4 };
   enum { gje3DimX_37      =  5 };
   enum { gje3DimX_38      =  4 };
   enum { gje3DimX_39      =  4 };
   enum { gje3DimX_40      =  4 };
   enum { gje3DimX_41      =  6 };
   enum { gje3DimX_42      =  6 };
   enum { gje3DimX_43      =  5 };
   enum { gje3DimX_44      =  4 };
   enum { gje3DimX_45      =  7 };
   enum { gje3DimX_46      =  6 };
   enum { gje3DimX_47      =  8 };
   enum { gje3DimX_48      =  8 };
   enum { gje3DimX_49      =  8 };
   enum { gje3DimX_50      =  4 };
   enum { gje3DimX_51      =  5 };
   enum { gje3DimX_52      =  4 };
   enum { gje3DimX_53      =  5 };
   enum { gje3DimX_54      =  6 };
   enum { gje3DimX_55      =  7 };
   enum { gje3DimX_56      =  9 };
   enum { gje3DimX_57      =  9 };
   enum { gje3DimX_58      = 10 };
   enum { gje3DimX_59      =  7 };
   enum { gje3DimX_60      =  8 };
   enum { gje3DimX_61      =  7 };
   enum { gje3DimX_62      =  7 };
   enum { gje3DimX_63      =  7 };
   enum { gje3DimX_64      =  8 };
   enum { gje3DimX_65      =  8 };
   enum { gje3DimX_66      =  8 };
   enum { gje3DimX_67      =  8 };
   enum { gje3DimX_68      =  8 };
   enum { gje3DimX_69      =  5 };
   enum { gje3DimX_70      =  6 };
   enum { gje3DimX_71      =  7 };
   enum { gje3DimX_72      =  9 };
   enum { gje3DimX_73      =  9 };
   enum { gje3DimX_74      =  6 };
   enum { gje3DimX_75      =  7 };
   enum { gje3DimX_76      =  7 };
   enum { gje3DimX_77      =  7 };

   enum { gje3Pad_00       =  0 };
   enum { gje3Pad_01       =  0 };
   enum { gje3Pad_02       =  0 };
   enum { gje3Pad_03       =  0 };
   enum { gje3Pad_04       =  0 };
   enum { gje3Pad_05       =  0 };
   enum { gje3Pad_06       =  0 };
   enum { gje3Pad_07       =  0 };
   enum { gje3Pad_08       =  4 };
   enum { gje3Pad_09       =  4 };
   enum { gje3Pad_10       =  0 };
   enum { gje3Pad_11       =  0 };
   enum { gje3Pad_12       =  0 };
   enum { gje3Pad_13       =  0 };
   enum { gje3Pad_14       =  0 };
   enum { gje3Pad_15       =  4 };
   enum { gje3Pad_16       =  4 };
   enum { gje3Pad_17       =  2 };
   enum { gje3Pad_18       =  1 };
   enum { gje3Pad_19       =  0 };
   enum { gje3Pad_20       =  0 };
   enum { gje3Pad_21       =  0 };
   enum { gje3Pad_22       =  0 };
   enum { gje3Pad_23       =  0 };
   enum { gje3Pad_24       =  4 };
   enum { gje3Pad_25       =  0 };
   enum { gje3Pad_26       =  0 };
   enum { gje3Pad_27       =  0 };
   enum { gje3Pad_28       =  0 };
   enum { gje3Pad_29       =  1 };
   enum { gje3Pad_30       =  0 };
   enum { gje3Pad_31       =  4 };
   enum { gje3Pad_32       =  3 };
   enum { gje3Pad_33       =  2 };
   enum { gje3Pad_34       =  2 };
   enum { gje3Pad_35       =  0 };
   enum { gje3Pad_36       =  0 };
   enum { gje3Pad_37       =  0 };
   enum { gje3Pad_38       =  0 };
   enum { gje3Pad_39       =  0 };
   enum { gje3Pad_40       =  4 };
   enum { gje3Pad_41       =  2 };
   enum { gje3Pad_42       =  0 };
   enum { gje3Pad_43       =  0 };
   enum { gje3Pad_44       =  0 };
   enum { gje3Pad_45       =  0 };
   enum { gje3Pad_46       =  0 };
   enum { gje3Pad_47       =  0 };
   enum { gje3Pad_48       =  1 };
   enum { gje3Pad_49       =  0 };
   enum { gje3Pad_50       =  2 };
   enum { gje3Pad_51       =  2 };
   enum { gje3Pad_52       =  0 };
   enum { gje3Pad_53       =  0 };
   enum { gje3Pad_54       =  0 };
   enum { gje3Pad_55       =  0 };
   enum { gje3Pad_56       =  1 };
   enum { gje3Pad_57       =  0 };
   enum { gje3Pad_58       =  0 };
   enum { gje3Pad_59       =  0 };
   enum { gje3Pad_60       =  0 };
   enum { gje3Pad_61       =  0 };
   enum { gje3Pad_62       =  0 };
   enum { gje3Pad_63       =  0 };
   enum { gje3Pad_64       =  2 };
   enum { gje3Pad_65       =  0 };
   enum { gje3Pad_66       =  0 };
   enum { gje3Pad_67       =  0 };
   enum { gje3Pad_68       =  4 };
   enum { gje3Pad_69       =  0 };
   enum { gje3Pad_70       =  0 };
   enum { gje3Pad_71       =  0 };
   enum { gje3Pad_72       =  1 };
   enum { gje3Pad_73       =  0 };
   enum { gje3Pad_74       =  0 };
   enum { gje3Pad_75       =  0 };
   enum { gje3Pad_76       =  0 };
   enum { gje3Pad_77       =  0 };

   enum { gje3SrchThrd_00  = -1 };
   enum { gje3SrchThrd_01  = -1 };
   enum { gje3SrchThrd_02  =  1 };
   enum { gje3SrchThrd_03  =  2 };    
   enum { gje3SrchThrd_04  =  2 };
   enum { gje3SrchThrd_05  =  2 };
   enum { gje3SrchThrd_06  =  2 };
   enum { gje3SrchThrd_07  =  2 };
   enum { gje3SrchThrd_08  =  2 };
   enum { gje3SrchThrd_09  =  2 };
   enum { gje3SrchThrd_10  =  2 };
   enum { gje3SrchThrd_11  =  2 };
   enum { gje3SrchThrd_12  =  3 };
   enum { gje3SrchThrd_13  =  3 };
   enum { gje3SrchThrd_14  =  3 };
   enum { gje3SrchThrd_15  =  3 };
   enum { gje3SrchThrd_16  =  3 };
   enum { gje3SrchThrd_17  =  3 };
   enum { gje3SrchThrd_18  =  3 };
   enum { gje3SrchThrd_19  =  3 };
   enum { gje3SrchThrd_20  =  3 };
   enum { gje3SrchThrd_21  =  3 };
   enum { gje3SrchThrd_22  =  3 };
   enum { gje3SrchThrd_23  =  3 };
   enum { gje3SrchThrd_24  =  3 };
   enum { gje3SrchThrd_25  =  3 };
   enum { gje3SrchThrd_26  =  3 };
   enum { gje3SrchThrd_27  =  3 };
   enum { gje3SrchThrd_28  =  3 };
   enum { gje3SrchThrd_29  =  4 };
   enum { gje3SrchThrd_30  =  4 };
   enum { gje3SrchThrd_31  =  4 };
   enum { gje3SrchThrd_32  =  4 };
   enum { gje3SrchThrd_33  =  4 };
   enum { gje3SrchThrd_34  =  4 };
   enum { gje3SrchThrd_35  =  4 };
   enum { gje3SrchThrd_36  =  4 };
   enum { gje3SrchThrd_37  =  4 };
   enum { gje3SrchThrd_38  =  4 };
   enum { gje3SrchThrd_39  =  4 };
   enum { gje3SrchThrd_40  =  4 };
   enum { gje3SrchThrd_41  =  4 };
   enum { gje3SrchThrd_42  =  4 };
   enum { gje3SrchThrd_43  =  4 };
   enum { gje3SrchThrd_44  =  4 };
   enum { gje3SrchThrd_45  =  4 };
   enum { gje3SrchThrd_46  =  4 };
   enum { gje3SrchThrd_47  =  4 };
   enum { gje3SrchThrd_48  =  4 };
   enum { gje3SrchThrd_49  =  4 };
   enum { gje3SrchThrd_50  =  4 };
   enum { gje3SrchThrd_51  =  4 };
   enum { gje3SrchThrd_52  =  4 };
   enum { gje3SrchThrd_53  =  4 };
   enum { gje3SrchThrd_54  =  5 };
   enum { gje3SrchThrd_55  =  6 };
   enum { gje3SrchThrd_56  =  6 };
   enum { gje3SrchThrd_57  =  6 };
   enum { gje3SrchThrd_58  =  6 };
   enum { gje3SrchThrd_59  =  6 };
   enum { gje3SrchThrd_60  =  6 };
   enum { gje3SrchThrd_61  =  6 };
   enum { gje3SrchThrd_62  =  6 };
   enum { gje3SrchThrd_63  =  6 };
   enum { gje3SrchThrd_64  =  6 };
   enum { gje3SrchThrd_65  =  6 };
   enum { gje3SrchThrd_66  =  6 };
   enum { gje3SrchThrd_67  =  6 };
   enum { gje3SrchThrd_68  =  6 };
   enum { gje3SrchThrd_69  =  6 };
   enum { gje3SrchThrd_70  =  6 };
   enum { gje3SrchThrd_71  =  6 };
   enum { gje3SrchThrd_72  =  6 };
   enum { gje3SrchThrd_73  =  6 };
   enum { gje3SrchThrd_74  =  6 };
   enum { gje3SrchThrd_75  =  6 };
   enum { gje3SrchThrd_76  =  6 };
   enum { gje3SrchThrd_77  =  6 };
};

template<> class config<cuDoubleComplex,ARCH_SM20> {
public:
   enum { gje3MinDim       =  2 };
   enum { gje3MaxDim       = 55 };
   enum { gje3MinBlks    =    1 };
   enum { gje3MaxThrds   = 1152 }; /* sm_2x, 28 registers per thread */

   enum { gje3DimX_00      = -1 };
   enum { gje3DimX_01      = -1 };
   enum { gje3DimX_02      =  2 };
   enum { gje3DimX_03      =  3 };
   enum { gje3DimX_04      =  4 };
   enum { gje3DimX_05      =  5 };
   enum { gje3DimX_06      =  5 };
   enum { gje3DimX_07      =  4 };
   enum { gje3DimX_08      =  8 };
   enum { gje3DimX_09      =  3 };
   enum { gje3DimX_10      =  6 };
   enum { gje3DimX_11      =  6 };
   enum { gje3DimX_12      =  4 };
   enum { gje3DimX_13      =  5 };
   enum { gje3DimX_14      =  4 };
   enum { gje3DimX_15      =  2 };
   enum { gje3DimX_16      =  4 };
   enum { gje3DimX_17      =  3 };
   enum { gje3DimX_18      =  4 };
   enum { gje3DimX_19      =  3 };
   enum { gje3DimX_20      =  4 };
   enum { gje3DimX_21      =  3 };
   enum { gje3DimX_22      =  4 };
   enum { gje3DimX_23      =  4 };
   enum { gje3DimX_24      =  8 };
   enum { gje3DimX_25      =  5 };
   enum { gje3DimX_26      =  4 };
   enum { gje3DimX_27      =  3 };
   enum { gje3DimX_28      =  8 };
   enum { gje3DimX_29      =  5 };
   enum { gje3DimX_30      =  6 };
   enum { gje3DimX_31      =  7 };
   enum { gje3DimX_32      =  8 };
   enum { gje3DimX_33      =  8 };
   enum { gje3DimX_34      =  8 };
   enum { gje3DimX_35      =  8 };
   enum { gje3DimX_36      =  8 };
   enum { gje3DimX_37      =  5 };
   enum { gje3DimX_38      =  6 };
   enum { gje3DimX_39      =  8 };
   enum { gje3DimX_40      =  8 };
   enum { gje3DimX_41      =  8 };
   enum { gje3DimX_42      =  8 };
   enum { gje3DimX_43      =  8 };
   enum { gje3DimX_44      =  8 };
   enum { gje3DimX_45      =  8 };
   enum { gje3DimX_46      =  8 };
   enum { gje3DimX_47      =  8 };
   enum { gje3DimX_48      =  8 };
   enum { gje3DimX_49      =  8 };
   enum { gje3DimX_50      =  8 };
   enum { gje3DimX_51      =  8 };
   enum { gje3DimX_52      =  8 };
   enum { gje3DimX_53      =  8 };
   enum { gje3DimX_54      =  6 };
   enum { gje3DimX_55      =  8 };
   enum { gje3DimX_56      = -1 };
   enum { gje3DimX_57      = -1 };
   enum { gje3DimX_58      = -1 };
   enum { gje3DimX_59      = -1 };
   enum { gje3DimX_60      = -1 };
   enum { gje3DimX_61      = -1 };
   enum { gje3DimX_62      = -1 };
   enum { gje3DimX_63      = -1 };
   enum { gje3DimX_64      = -1 };
   enum { gje3DimX_65      = -1 };
   enum { gje3DimX_66      = -1 };
   enum { gje3DimX_67      = -1 };
   enum { gje3DimX_68      = -1 };
   enum { gje3DimX_69      = -1 };
   enum { gje3DimX_70      = -1 };
   enum { gje3DimX_71      = -1 };
   enum { gje3DimX_72      = -1 };
   enum { gje3DimX_73      = -1 };
   enum { gje3DimX_74      = -1 };
   enum { gje3DimX_75      = -1 };
   enum { gje3DimX_76      = -1 };
   enum { gje3DimX_77      = -1 };

   enum { gje3Pad_00       =  0 };
   enum { gje3Pad_01       =  0 };
   enum { gje3Pad_02       =  0 };
   enum { gje3Pad_03       =  0 };
   enum { gje3Pad_04       =  0 };
   enum { gje3Pad_05       =  0 };
   enum { gje3Pad_06       =  0 };
   enum { gje3Pad_07       =  4 };
   enum { gje3Pad_08       =  2 };
   enum { gje3Pad_09       =  2 };
   enum { gje3Pad_10       =  4 };
   enum { gje3Pad_11       =  3 };
   enum { gje3Pad_12       =  2 };
   enum { gje3Pad_13       =  0 };
   enum { gje3Pad_14       =  0 };
   enum { gje3Pad_15       =  0 };
   enum { gje3Pad_16       =  2 };
   enum { gje3Pad_17       =  2 };
   enum { gje3Pad_18       =  0 };
   enum { gje3Pad_19       =  0 };
   enum { gje3Pad_20       =  0 };
   enum { gje3Pad_21       =  0 };
   enum { gje3Pad_22       =  0 };
   enum { gje3Pad_23       =  0 };
   enum { gje3Pad_24       =  1 };
   enum { gje3Pad_25       =  4 };
   enum { gje3Pad_26       =  0 };
   enum { gje3Pad_27       =  0 };
   enum { gje3Pad_28       =  0 };
   enum { gje3Pad_29       =  0 };
   enum { gje3Pad_30       =  0 };
   enum { gje3Pad_31       =  0 };
   enum { gje3Pad_32       =  1 };
   enum { gje3Pad_33       =  0 };
   enum { gje3Pad_34       =  0 };
   enum { gje3Pad_35       =  0 };
   enum { gje3Pad_36       =  0 };
   enum { gje3Pad_37       =  0 };
   enum { gje3Pad_38       =  0 };
   enum { gje3Pad_39       =  0 };
   enum { gje3Pad_40       =  1 };
   enum { gje3Pad_41       =  0 };
   enum { gje3Pad_42       =  0 };
   enum { gje3Pad_43       =  0 };
   enum { gje3Pad_44       =  0 };
   enum { gje3Pad_45       =  0 };
   enum { gje3Pad_46       =  0 };
   enum { gje3Pad_47       =  0 };
   enum { gje3Pad_48       =  1 };
   enum { gje3Pad_49       =  0 };
   enum { gje3Pad_50       =  0 };
   enum { gje3Pad_51       =  0 };
   enum { gje3Pad_52       =  0 };
   enum { gje3Pad_53       =  0 };
   enum { gje3Pad_54       =  0 };
   enum { gje3Pad_55       =  0 };
   enum { gje3Pad_56       =  0 };
   enum { gje3Pad_57       =  0 };
   enum { gje3Pad_58       =  0 };
   enum { gje3Pad_59       =  0 };
   enum { gje3Pad_60       =  0 };
   enum { gje3Pad_61       =  0 };
   enum { gje3Pad_62       =  0 };
   enum { gje3Pad_63       =  0 };
   enum { gje3Pad_64       =  0 };
   enum { gje3Pad_65       =  0 };
   enum { gje3Pad_66       =  0 };
   enum { gje3Pad_67       =  0 };
   enum { gje3Pad_68       =  0 };
   enum { gje3Pad_69       =  0 };
   enum { gje3Pad_70       =  0 };
   enum { gje3Pad_71       =  0 };
   enum { gje3Pad_72       =  0 };
   enum { gje3Pad_73       =  0 };
   enum { gje3Pad_74       =  0 };
   enum { gje3Pad_75       =  0 };
   enum { gje3Pad_76       =  0 };
   enum { gje3Pad_77       =  0 };

   enum { gje3SrchThrd_00  = -1 };
   enum { gje3SrchThrd_01  = -1 };
   enum { gje3SrchThrd_02  =  1 };
   enum { gje3SrchThrd_03  =  2 };    
   enum { gje3SrchThrd_04  =  2 };
   enum { gje3SrchThrd_05  =  2 };
   enum { gje3SrchThrd_06  =  2 };
   enum { gje3SrchThrd_07  =  2 };
   enum { gje3SrchThrd_08  =  3 };
   enum { gje3SrchThrd_09  =  3 };
   enum { gje3SrchThrd_10  =  3 };
   enum { gje3SrchThrd_11  =  3 };
   enum { gje3SrchThrd_12  =  3 };
   enum { gje3SrchThrd_13  =  3 };
   enum { gje3SrchThrd_14  =  3 };
   enum { gje3SrchThrd_15  =  3 };
   enum { gje3SrchThrd_16  =  3 };
   enum { gje3SrchThrd_17  =  3 };
   enum { gje3SrchThrd_18  =  3 };
   enum { gje3SrchThrd_19  =  3 };
   enum { gje3SrchThrd_20  =  3 };
   enum { gje3SrchThrd_21  =  4 };
   enum { gje3SrchThrd_22  =  4 };
   enum { gje3SrchThrd_23  =  4 };
   enum { gje3SrchThrd_24  =  4 };
   enum { gje3SrchThrd_25  =  4 };
   enum { gje3SrchThrd_26  =  4 };
   enum { gje3SrchThrd_27  =  4 };
   enum { gje3SrchThrd_28  =  4 };
   enum { gje3SrchThrd_29  =  4 };
   enum { gje3SrchThrd_30  =  4 };
   enum { gje3SrchThrd_31  =  4 };
   enum { gje3SrchThrd_32  =  4 };
   enum { gje3SrchThrd_33  =  4 };
   enum { gje3SrchThrd_34  =  4 };
   enum { gje3SrchThrd_35  =  4 };
   enum { gje3SrchThrd_36  =  4 };
   enum { gje3SrchThrd_37  =  6 };
   enum { gje3SrchThrd_38  =  6 };
   enum { gje3SrchThrd_39  =  6 };
   enum { gje3SrchThrd_40  =  6 };
   enum { gje3SrchThrd_41  =  6 };
   enum { gje3SrchThrd_42  =  6 };
   enum { gje3SrchThrd_43  =  6 };
   enum { gje3SrchThrd_44  =  6 };
   enum { gje3SrchThrd_45  =  6 };
   enum { gje3SrchThrd_46  =  7 };
   enum { gje3SrchThrd_47  =  7 };
   enum { gje3SrchThrd_48  =  7 };
   enum { gje3SrchThrd_49  =  7 };
   enum { gje3SrchThrd_50  =  7 };
   enum { gje3SrchThrd_51  =  7 };
   enum { gje3SrchThrd_52  =  7 };
   enum { gje3SrchThrd_53  =  7 };
   enum { gje3SrchThrd_54  =  7 };
   enum { gje3SrchThrd_55  =  7 };
   enum { gje3SrchThrd_56  = -1 };
   enum { gje3SrchThrd_57  = -1 };
   enum { gje3SrchThrd_58  = -1 };
   enum { gje3SrchThrd_59  = -1 };
   enum { gje3SrchThrd_60  = -1 };
   enum { gje3SrchThrd_61  = -1 };
   enum { gje3SrchThrd_62  = -1 };
   enum { gje3SrchThrd_63  = -1 };
   enum { gje3SrchThrd_64  = -1 };
   enum { gje3SrchThrd_65  = -1 };
   enum { gje3SrchThrd_66  = -1 };
   enum { gje3SrchThrd_67  = -1 };
   enum { gje3SrchThrd_68  = -1 };
   enum { gje3SrchThrd_69  = -1 };
   enum { gje3SrchThrd_70  = -1 };
   enum { gje3SrchThrd_71  = -1 };
   enum { gje3SrchThrd_72  = -1 };
   enum { gje3SrchThrd_73  = -1 };
   enum { gje3SrchThrd_74  = -1 };
   enum { gje3SrchThrd_75  = -1 };
   enum { gje3SrchThrd_76  = -1 };
   enum { gje3SrchThrd_77  = -1 };
};


// cuComplex and float are just copies of the double config (however float should be something different)
template<> class config<cuComplex,ARCH_SM20> {
public:
   enum { gje3MinDim       =  2 };
   enum { gje3MaxDim       = 77 };
   enum { gje3MinBlks    =    1 };
   enum { gje3MaxThrds   = 1408 }; /* sm_2x, 23 registers per thread */

   enum { gje3DimX_00      = -1 };
   enum { gje3DimX_01      = -1 };
   enum { gje3DimX_02      =  2 };
   enum { gje3DimX_03      =  3 };
   enum { gje3DimX_04      =  4 };
   enum { gje3DimX_05      =  5 };
   enum { gje3DimX_06      =  5 };
   enum { gje3DimX_07      =  7 };
   enum { gje3DimX_08      =  4 };
   enum { gje3DimX_09      =  3 };
   enum { gje3DimX_10      =  6 };
   enum { gje3DimX_11      =  5 };
   enum { gje3DimX_12      =  4 };
   enum { gje3DimX_13      =  4 };
   enum { gje3DimX_14      =  4 };
   enum { gje3DimX_15      =  4 };
   enum { gje3DimX_16      =  4 };
   enum { gje3DimX_17      =  3 };
   enum { gje3DimX_18      =  3 };
   enum { gje3DimX_19      =  3 };
   enum { gje3DimX_20      =  4 };
   enum { gje3DimX_21      =  3 };
   enum { gje3DimX_22      =  4 };
   enum { gje3DimX_23      =  4 };
   enum { gje3DimX_24      =  4 };
   enum { gje3DimX_25      =  5 };
   enum { gje3DimX_26      =  2 };
   enum { gje3DimX_27      =  3 };
   enum { gje3DimX_28      =  4 };
   enum { gje3DimX_29      =  3 };
   enum { gje3DimX_30      =  3 };
   enum { gje3DimX_31      =  3 };
   enum { gje3DimX_32      =  3 };
   enum { gje3DimX_33      =  3 };
   enum { gje3DimX_34      =  4 };
   enum { gje3DimX_35      =  3 };
   enum { gje3DimX_36      =  4 };
   enum { gje3DimX_37      =  5 };
   enum { gje3DimX_38      =  4 };
   enum { gje3DimX_39      =  4 };
   enum { gje3DimX_40      =  4 };
   enum { gje3DimX_41      =  6 };
   enum { gje3DimX_42      =  6 };
   enum { gje3DimX_43      =  5 };
   enum { gje3DimX_44      =  4 };
   enum { gje3DimX_45      =  7 };
   enum { gje3DimX_46      =  6 };
   enum { gje3DimX_47      =  8 };
   enum { gje3DimX_48      =  8 };
   enum { gje3DimX_49      =  8 };
   enum { gje3DimX_50      =  4 };
   enum { gje3DimX_51      =  5 };
   enum { gje3DimX_52      =  4 };
   enum { gje3DimX_53      =  5 };
   enum { gje3DimX_54      =  6 };
   enum { gje3DimX_55      =  7 };
   enum { gje3DimX_56      =  9 };
   enum { gje3DimX_57      =  9 };
   enum { gje3DimX_58      = 10 };
   enum { gje3DimX_59      =  7 };
   enum { gje3DimX_60      =  8 };
   enum { gje3DimX_61      =  7 };
   enum { gje3DimX_62      =  7 };
   enum { gje3DimX_63      =  7 };
   enum { gje3DimX_64      =  8 };
   enum { gje3DimX_65      =  8 };
   enum { gje3DimX_66      =  8 };
   enum { gje3DimX_67      =  8 };
   enum { gje3DimX_68      =  8 };
   enum { gje3DimX_69      =  5 };
   enum { gje3DimX_70      =  6 };
   enum { gje3DimX_71      =  7 };
   enum { gje3DimX_72      =  9 };
   enum { gje3DimX_73      =  9 };
   enum { gje3DimX_74      =  6 };
   enum { gje3DimX_75      =  7 };
   enum { gje3DimX_76      =  7 };
   enum { gje3DimX_77      =  7 };

   enum { gje3Pad_00       =  0 };
   enum { gje3Pad_01       =  0 };
   enum { gje3Pad_02       =  0 };
   enum { gje3Pad_03       =  0 };
   enum { gje3Pad_04       =  0 };
   enum { gje3Pad_05       =  0 };
   enum { gje3Pad_06       =  0 };
   enum { gje3Pad_07       =  0 };
   enum { gje3Pad_08       =  4 };
   enum { gje3Pad_09       =  4 };
   enum { gje3Pad_10       =  0 };
   enum { gje3Pad_11       =  0 };
   enum { gje3Pad_12       =  0 };
   enum { gje3Pad_13       =  0 };
   enum { gje3Pad_14       =  0 };
   enum { gje3Pad_15       =  4 };
   enum { gje3Pad_16       =  4 };
   enum { gje3Pad_17       =  2 };
   enum { gje3Pad_18       =  1 };
   enum { gje3Pad_19       =  0 };
   enum { gje3Pad_20       =  0 };
   enum { gje3Pad_21       =  0 };
   enum { gje3Pad_22       =  0 };
   enum { gje3Pad_23       =  0 };
   enum { gje3Pad_24       =  4 };
   enum { gje3Pad_25       =  0 };
   enum { gje3Pad_26       =  0 };
   enum { gje3Pad_27       =  0 };
   enum { gje3Pad_28       =  0 };
   enum { gje3Pad_29       =  1 };
   enum { gje3Pad_30       =  0 };
   enum { gje3Pad_31       =  4 };
   enum { gje3Pad_32       =  3 };
   enum { gje3Pad_33       =  2 };
   enum { gje3Pad_34       =  2 };
   enum { gje3Pad_35       =  0 };
   enum { gje3Pad_36       =  0 };
   enum { gje3Pad_37       =  0 };
   enum { gje3Pad_38       =  0 };
   enum { gje3Pad_39       =  0 };
   enum { gje3Pad_40       =  4 };
   enum { gje3Pad_41       =  2 };
   enum { gje3Pad_42       =  0 };
   enum { gje3Pad_43       =  0 };
   enum { gje3Pad_44       =  0 };
   enum { gje3Pad_45       =  0 };
   enum { gje3Pad_46       =  0 };
   enum { gje3Pad_47       =  0 };
   enum { gje3Pad_48       =  1 };
   enum { gje3Pad_49       =  0 };
   enum { gje3Pad_50       =  2 };
   enum { gje3Pad_51       =  2 };
   enum { gje3Pad_52       =  0 };
   enum { gje3Pad_53       =  0 };
   enum { gje3Pad_54       =  0 };
   enum { gje3Pad_55       =  0 };
   enum { gje3Pad_56       =  1 };
   enum { gje3Pad_57       =  0 };
   enum { gje3Pad_58       =  0 };
   enum { gje3Pad_59       =  0 };
   enum { gje3Pad_60       =  0 };
   enum { gje3Pad_61       =  0 };
   enum { gje3Pad_62       =  0 };
   enum { gje3Pad_63       =  0 };
   enum { gje3Pad_64       =  2 };
   enum { gje3Pad_65       =  0 };
   enum { gje3Pad_66       =  0 };
   enum { gje3Pad_67       =  0 };
   enum { gje3Pad_68       =  4 };
   enum { gje3Pad_69       =  0 };
   enum { gje3Pad_70       =  0 };
   enum { gje3Pad_71       =  0 };
   enum { gje3Pad_72       =  1 };
   enum { gje3Pad_73       =  0 };
   enum { gje3Pad_74       =  0 };
   enum { gje3Pad_75       =  0 };
   enum { gje3Pad_76       =  0 };
   enum { gje3Pad_77       =  0 };

   enum { gje3SrchThrd_00  = -1 };
   enum { gje3SrchThrd_01  = -1 };
   enum { gje3SrchThrd_02  =  1 };
   enum { gje3SrchThrd_03  =  2 };    
   enum { gje3SrchThrd_04  =  2 };
   enum { gje3SrchThrd_05  =  2 };
   enum { gje3SrchThrd_06  =  2 };
   enum { gje3SrchThrd_07  =  2 };
   enum { gje3SrchThrd_08  =  2 };
   enum { gje3SrchThrd_09  =  2 };
   enum { gje3SrchThrd_10  =  2 };
   enum { gje3SrchThrd_11  =  2 };
   enum { gje3SrchThrd_12  =  3 };
   enum { gje3SrchThrd_13  =  3 };
   enum { gje3SrchThrd_14  =  3 };
   enum { gje3SrchThrd_15  =  3 };
   enum { gje3SrchThrd_16  =  3 };
   enum { gje3SrchThrd_17  =  3 };
   enum { gje3SrchThrd_18  =  3 };
   enum { gje3SrchThrd_19  =  3 };
   enum { gje3SrchThrd_20  =  3 };
   enum { gje3SrchThrd_21  =  3 };
   enum { gje3SrchThrd_22  =  3 };
   enum { gje3SrchThrd_23  =  3 };
   enum { gje3SrchThrd_24  =  3 };
   enum { gje3SrchThrd_25  =  3 };
   enum { gje3SrchThrd_26  =  3 };
   enum { gje3SrchThrd_27  =  3 };
   enum { gje3SrchThrd_28  =  3 };
   enum { gje3SrchThrd_29  =  4 };
   enum { gje3SrchThrd_30  =  4 };
   enum { gje3SrchThrd_31  =  4 };
   enum { gje3SrchThrd_32  =  4 };
   enum { gje3SrchThrd_33  =  4 };
   enum { gje3SrchThrd_34  =  4 };
   enum { gje3SrchThrd_35  =  4 };
   enum { gje3SrchThrd_36  =  4 };
   enum { gje3SrchThrd_37  =  4 };
   enum { gje3SrchThrd_38  =  4 };
   enum { gje3SrchThrd_39  =  4 };
   enum { gje3SrchThrd_40  =  4 };
   enum { gje3SrchThrd_41  =  4 };
   enum { gje3SrchThrd_42  =  4 };
   enum { gje3SrchThrd_43  =  4 };
   enum { gje3SrchThrd_44  =  4 };
   enum { gje3SrchThrd_45  =  4 };
   enum { gje3SrchThrd_46  =  4 };
   enum { gje3SrchThrd_47  =  4 };
   enum { gje3SrchThrd_48  =  4 };
   enum { gje3SrchThrd_49  =  4 };
   enum { gje3SrchThrd_50  =  4 };
   enum { gje3SrchThrd_51  =  4 };
   enum { gje3SrchThrd_52  =  4 };
   enum { gje3SrchThrd_53  =  4 };
   enum { gje3SrchThrd_54  =  5 };
   enum { gje3SrchThrd_55  =  6 };
   enum { gje3SrchThrd_56  =  6 };
   enum { gje3SrchThrd_57  =  6 };
   enum { gje3SrchThrd_58  =  6 };
   enum { gje3SrchThrd_59  =  6 };
   enum { gje3SrchThrd_60  =  6 };
   enum { gje3SrchThrd_61  =  6 };
   enum { gje3SrchThrd_62  =  6 };
   enum { gje3SrchThrd_63  =  6 };
   enum { gje3SrchThrd_64  =  6 };
   enum { gje3SrchThrd_65  =  6 };
   enum { gje3SrchThrd_66  =  6 };
   enum { gje3SrchThrd_67  =  6 };
   enum { gje3SrchThrd_68  =  6 };
   enum { gje3SrchThrd_69  =  6 };
   enum { gje3SrchThrd_70  =  6 };
   enum { gje3SrchThrd_71  =  6 };
   enum { gje3SrchThrd_72  =  6 };
   enum { gje3SrchThrd_73  =  6 };
   enum { gje3SrchThrd_74  =  6 };
   enum { gje3SrchThrd_75  =  6 };
   enum { gje3SrchThrd_76  =  6 };
   enum { gje3SrchThrd_77  =  6 };
};

template<> class config<float,ARCH_SM20> {
public:
   enum { gje3MinDim       =  2 };
   enum { gje3MaxDim       = 77 };
   enum { gje3MinBlks    =    1 };
   enum { gje3MaxThrds   = 1408 }; /* sm_2x, 23 registers per thread */

   enum { gje3DimX_00      = -1 };
   enum { gje3DimX_01      = -1 };
   enum { gje3DimX_02      =  2 };
   enum { gje3DimX_03      =  3 };
   enum { gje3DimX_04      =  4 };
   enum { gje3DimX_05      =  5 };
   enum { gje3DimX_06      =  5 };
   enum { gje3DimX_07      =  7 };
   enum { gje3DimX_08      =  4 };
   enum { gje3DimX_09      =  3 };
   enum { gje3DimX_10      =  6 };
   enum { gje3DimX_11      =  5 };
   enum { gje3DimX_12      =  4 };
   enum { gje3DimX_13      =  4 };
   enum { gje3DimX_14      =  4 };
   enum { gje3DimX_15      =  4 };
   enum { gje3DimX_16      =  4 };
   enum { gje3DimX_17      =  3 };
   enum { gje3DimX_18      =  3 };
   enum { gje3DimX_19      =  3 };
   enum { gje3DimX_20      =  4 };
   enum { gje3DimX_21      =  3 };
   enum { gje3DimX_22      =  4 };
   enum { gje3DimX_23      =  4 };
   enum { gje3DimX_24      =  4 };
   enum { gje3DimX_25      =  5 };
   enum { gje3DimX_26      =  2 };
   enum { gje3DimX_27      =  3 };
   enum { gje3DimX_28      =  4 };
   enum { gje3DimX_29      =  3 };
   enum { gje3DimX_30      =  3 };
   enum { gje3DimX_31      =  3 };
   enum { gje3DimX_32      =  3 };
   enum { gje3DimX_33      =  3 };
   enum { gje3DimX_34      =  4 };
   enum { gje3DimX_35      =  3 };
   enum { gje3DimX_36      =  4 };
   enum { gje3DimX_37      =  5 };
   enum { gje3DimX_38      =  4 };
   enum { gje3DimX_39      =  4 };
   enum { gje3DimX_40      =  4 };
   enum { gje3DimX_41      =  6 };
   enum { gje3DimX_42      =  6 };
   enum { gje3DimX_43      =  5 };
   enum { gje3DimX_44      =  4 };
   enum { gje3DimX_45      =  7 };
   enum { gje3DimX_46      =  6 };
   enum { gje3DimX_47      =  8 };
   enum { gje3DimX_48      =  8 };
   enum { gje3DimX_49      =  8 };
   enum { gje3DimX_50      =  4 };
   enum { gje3DimX_51      =  5 };
   enum { gje3DimX_52      =  4 };
   enum { gje3DimX_53      =  5 };
   enum { gje3DimX_54      =  6 };
   enum { gje3DimX_55      =  7 };
   enum { gje3DimX_56      =  9 };
   enum { gje3DimX_57      =  9 };
   enum { gje3DimX_58      = 10 };
   enum { gje3DimX_59      =  7 };
   enum { gje3DimX_60      =  8 };
   enum { gje3DimX_61      =  7 };
   enum { gje3DimX_62      =  7 };
   enum { gje3DimX_63      =  7 };
   enum { gje3DimX_64      =  8 };
   enum { gje3DimX_65      =  8 };
   enum { gje3DimX_66      =  8 };
   enum { gje3DimX_67      =  8 };
   enum { gje3DimX_68      =  8 };
   enum { gje3DimX_69      =  5 };
   enum { gje3DimX_70      =  6 };
   enum { gje3DimX_71      =  7 };
   enum { gje3DimX_72      =  9 };
   enum { gje3DimX_73      =  9 };
   enum { gje3DimX_74      =  6 };
   enum { gje3DimX_75      =  7 };
   enum { gje3DimX_76      =  7 };
   enum { gje3DimX_77      =  7 };

   enum { gje3Pad_00       =  0 };
   enum { gje3Pad_01       =  0 };
   enum { gje3Pad_02       =  0 };
   enum { gje3Pad_03       =  0 };
   enum { gje3Pad_04       =  0 };
   enum { gje3Pad_05       =  0 };
   enum { gje3Pad_06       =  0 };
   enum { gje3Pad_07       =  0 };
   enum { gje3Pad_08       =  4 };
   enum { gje3Pad_09       =  4 };
   enum { gje3Pad_10       =  0 };
   enum { gje3Pad_11       =  0 };
   enum { gje3Pad_12       =  0 };
   enum { gje3Pad_13       =  0 };
   enum { gje3Pad_14       =  0 };
   enum { gje3Pad_15       =  4 };
   enum { gje3Pad_16       =  4 };
   enum { gje3Pad_17       =  2 };
   enum { gje3Pad_18       =  1 };
   enum { gje3Pad_19       =  0 };
   enum { gje3Pad_20       =  0 };
   enum { gje3Pad_21       =  0 };
   enum { gje3Pad_22       =  0 };
   enum { gje3Pad_23       =  0 };
   enum { gje3Pad_24       =  4 };
   enum { gje3Pad_25       =  0 };
   enum { gje3Pad_26       =  0 };
   enum { gje3Pad_27       =  0 };
   enum { gje3Pad_28       =  0 };
   enum { gje3Pad_29       =  1 };
   enum { gje3Pad_30       =  0 };
   enum { gje3Pad_31       =  4 };
   enum { gje3Pad_32       =  3 };
   enum { gje3Pad_33       =  2 };
   enum { gje3Pad_34       =  2 };
   enum { gje3Pad_35       =  0 };
   enum { gje3Pad_36       =  0 };
   enum { gje3Pad_37       =  0 };
   enum { gje3Pad_38       =  0 };
   enum { gje3Pad_39       =  0 };
   enum { gje3Pad_40       =  4 };
   enum { gje3Pad_41       =  2 };
   enum { gje3Pad_42       =  0 };
   enum { gje3Pad_43       =  0 };
   enum { gje3Pad_44       =  0 };
   enum { gje3Pad_45       =  0 };
   enum { gje3Pad_46       =  0 };
   enum { gje3Pad_47       =  0 };
   enum { gje3Pad_48       =  1 };
   enum { gje3Pad_49       =  0 };
   enum { gje3Pad_50       =  2 };
   enum { gje3Pad_51       =  2 };
   enum { gje3Pad_52       =  0 };
   enum { gje3Pad_53       =  0 };
   enum { gje3Pad_54       =  0 };
   enum { gje3Pad_55       =  0 };
   enum { gje3Pad_56       =  1 };
   enum { gje3Pad_57       =  0 };
   enum { gje3Pad_58       =  0 };
   enum { gje3Pad_59       =  0 };
   enum { gje3Pad_60       =  0 };
   enum { gje3Pad_61       =  0 };
   enum { gje3Pad_62       =  0 };
   enum { gje3Pad_63       =  0 };
   enum { gje3Pad_64       =  2 };
   enum { gje3Pad_65       =  0 };
   enum { gje3Pad_66       =  0 };
   enum { gje3Pad_67       =  0 };
   enum { gje3Pad_68       =  4 };
   enum { gje3Pad_69       =  0 };
   enum { gje3Pad_70       =  0 };
   enum { gje3Pad_71       =  0 };
   enum { gje3Pad_72       =  1 };
   enum { gje3Pad_73       =  0 };
   enum { gje3Pad_74       =  0 };
   enum { gje3Pad_75       =  0 };
   enum { gje3Pad_76       =  0 };
   enum { gje3Pad_77       =  0 };

   enum { gje3SrchThrd_00  = -1 };
   enum { gje3SrchThrd_01  = -1 };
   enum { gje3SrchThrd_02  =  1 };
   enum { gje3SrchThrd_03  =  2 };    
   enum { gje3SrchThrd_04  =  2 };
   enum { gje3SrchThrd_05  =  2 };
   enum { gje3SrchThrd_06  =  2 };
   enum { gje3SrchThrd_07  =  2 };
   enum { gje3SrchThrd_08  =  2 };
   enum { gje3SrchThrd_09  =  2 };
   enum { gje3SrchThrd_10  =  2 };
   enum { gje3SrchThrd_11  =  2 };
   enum { gje3SrchThrd_12  =  3 };
   enum { gje3SrchThrd_13  =  3 };
   enum { gje3SrchThrd_14  =  3 };
   enum { gje3SrchThrd_15  =  3 };
   enum { gje3SrchThrd_16  =  3 };
   enum { gje3SrchThrd_17  =  3 };
   enum { gje3SrchThrd_18  =  3 };
   enum { gje3SrchThrd_19  =  3 };
   enum { gje3SrchThrd_20  =  3 };
   enum { gje3SrchThrd_21  =  3 };
   enum { gje3SrchThrd_22  =  3 };
   enum { gje3SrchThrd_23  =  3 };
   enum { gje3SrchThrd_24  =  3 };
   enum { gje3SrchThrd_25  =  3 };
   enum { gje3SrchThrd_26  =  3 };
   enum { gje3SrchThrd_27  =  3 };
   enum { gje3SrchThrd_28  =  3 };
   enum { gje3SrchThrd_29  =  4 };
   enum { gje3SrchThrd_30  =  4 };
   enum { gje3SrchThrd_31  =  4 };
   enum { gje3SrchThrd_32  =  4 };
   enum { gje3SrchThrd_33  =  4 };
   enum { gje3SrchThrd_34  =  4 };
   enum { gje3SrchThrd_35  =  4 };
   enum { gje3SrchThrd_36  =  4 };
   enum { gje3SrchThrd_37  =  4 };
   enum { gje3SrchThrd_38  =  4 };
   enum { gje3SrchThrd_39  =  4 };
   enum { gje3SrchThrd_40  =  4 };
   enum { gje3SrchThrd_41  =  4 };
   enum { gje3SrchThrd_42  =  4 };
   enum { gje3SrchThrd_43  =  4 };
   enum { gje3SrchThrd_44  =  4 };
   enum { gje3SrchThrd_45  =  4 };
   enum { gje3SrchThrd_46  =  4 };
   enum { gje3SrchThrd_47  =  4 };
   enum { gje3SrchThrd_48  =  4 };
   enum { gje3SrchThrd_49  =  4 };
   enum { gje3SrchThrd_50  =  4 };
   enum { gje3SrchThrd_51  =  4 };
   enum { gje3SrchThrd_52  =  4 };
   enum { gje3SrchThrd_53  =  4 };
   enum { gje3SrchThrd_54  =  5 };
   enum { gje3SrchThrd_55  =  6 };
   enum { gje3SrchThrd_56  =  6 };
   enum { gje3SrchThrd_57  =  6 };
   enum { gje3SrchThrd_58  =  6 };
   enum { gje3SrchThrd_59  =  6 };
   enum { gje3SrchThrd_60  =  6 };
   enum { gje3SrchThrd_61  =  6 };
   enum { gje3SrchThrd_62  =  6 };
   enum { gje3SrchThrd_63  =  6 };
   enum { gje3SrchThrd_64  =  6 };
   enum { gje3SrchThrd_65  =  6 };
   enum { gje3SrchThrd_66  =  6 };
   enum { gje3SrchThrd_67  =  6 };
   enum { gje3SrchThrd_68  =  6 };
   enum { gje3SrchThrd_69  =  6 };
   enum { gje3SrchThrd_70  =  6 };
   enum { gje3SrchThrd_71  =  6 };
   enum { gje3SrchThrd_72  =  6 };
   enum { gje3SrchThrd_73  =  6 };
   enum { gje3SrchThrd_74  =  6 };
   enum { gje3SrchThrd_75  =  6 };
   enum { gje3SrchThrd_76  =  6 };
   enum { gje3SrchThrd_77  =  6 };
};




/** ARCH_SM13 configs **/
template<> class config<double,ARCH_SM13> {
public:
   enum { gje3MinDim       =  2 };
   enum { gje3MaxDim       = 44 };
   enum { gje3MinBlks      =  1 };
   enum { gje3MaxThrds     =768 }; /* sm_13, 21 registers per thread */

   enum { gje3DimX_00      = -1 };
   enum { gje3DimX_01      = -1 };
   enum { gje3DimX_02      =  2 };
   enum { gje3DimX_03      =  3 };
   enum { gje3DimX_04      =  4 };
   enum { gje3DimX_05      =  3 };
   enum { gje3DimX_06      =  2 };
   enum { gje3DimX_07      =  2 };
   enum { gje3DimX_08      =  2 };
   enum { gje3DimX_09      =  3 };
   enum { gje3DimX_10      =  3 };
   enum { gje3DimX_11      =  2 };
   enum { gje3DimX_12      =  4 };
   enum { gje3DimX_13      =  2 };
   enum { gje3DimX_14      =  2 };
   enum { gje3DimX_15      =  2 };
   enum { gje3DimX_16      =  2 };
   enum { gje3DimX_17      =  2 };
   enum { gje3DimX_18      =  2 };
   enum { gje3DimX_19      =  3 };
   enum { gje3DimX_20      =  4 };
   enum { gje3DimX_21      =  3 };
   enum { gje3DimX_22      =  4 };
   enum { gje3DimX_23      =  2 };
   enum { gje3DimX_24      =  2 };
   enum { gje3DimX_25      =  5 };
   enum { gje3DimX_26      =  4 };
   enum { gje3DimX_27      =  4 };
   enum { gje3DimX_28      =  4 };
   enum { gje3DimX_29      =  5 };
   enum { gje3DimX_30      =  4 };
   enum { gje3DimX_31      =  2 };
   enum { gje3DimX_32      =  8 };
   enum { gje3DimX_33      =  7 };
   enum { gje3DimX_34      =  7 };
   enum { gje3DimX_35      =  7 };
   enum { gje3DimX_36      =  8 };
   enum { gje3DimX_37      =  8 };
   enum { gje3DimX_38      =  8 };
   enum { gje3DimX_39      =  8 };
   enum { gje3DimX_40      =  8 };
   enum { gje3DimX_41      =  7 };
   enum { gje3DimX_42      =  6 };
   enum { gje3DimX_43      =  8 };
   enum { gje3DimX_44      =  8 };
   enum { gje3DimX_45      = -1 };
   enum { gje3DimX_46      = -1 };
   enum { gje3DimX_47      = -1 };
   enum { gje3DimX_48      = -1 };
   enum { gje3DimX_49      = -1 };
   enum { gje3DimX_50      = -1 };
   enum { gje3DimX_51      = -1 };
   enum { gje3DimX_52      = -1 };
   enum { gje3DimX_53      = -1 };
   enum { gje3DimX_54      = -1 };
   enum { gje3DimX_55      = -1 };
   enum { gje3DimX_56      = -1 };
   enum { gje3DimX_57      = -1 };
   enum { gje3DimX_58      = -1 };
   enum { gje3DimX_59      = -1 };
   enum { gje3DimX_60      = -1 };
   enum { gje3DimX_61      = -1 };
   enum { gje3DimX_62      = -1 };
   enum { gje3DimX_63      = -1 };
   enum { gje3DimX_64      = -1 };
   enum { gje3DimX_65      = -1 };
   enum { gje3DimX_66      = -1 };
   enum { gje3DimX_67      = -1 };
   enum { gje3DimX_68      = -1 };
   enum { gje3DimX_69      = -1 };
   enum { gje3DimX_70      = -1 };
   enum { gje3DimX_71      = -1 };
   enum { gje3DimX_72      = -1 };
   enum { gje3DimX_73      = -1 };
   enum { gje3DimX_74      = -1 };
   enum { gje3DimX_75      = -1 };
   enum { gje3DimX_76      = -1 };
   enum { gje3DimX_77      = -1 };

   enum { gje3Pad_00       =  0 };
   enum { gje3Pad_01       =  0 };
   enum { gje3Pad_02       =  0 };
   enum { gje3Pad_03       =  0 };
   enum { gje3Pad_04       =  2 };
   enum { gje3Pad_05       =  0 };
   enum { gje3Pad_06       =  1 };
   enum { gje3Pad_07       =  4 };
   enum { gje3Pad_08       =  3 };
   enum { gje3Pad_09       =  2 };
   enum { gje3Pad_10       =  1 };
   enum { gje3Pad_11       =  2 };
   enum { gje3Pad_12       =  2 };
   enum { gje3Pad_13       =  2 };
   enum { gje3Pad_14       =  1 };
   enum { gje3Pad_15       =  0 };
   enum { gje3Pad_16       =  1 };
   enum { gje3Pad_17       =  0 };
   enum { gje3Pad_18       =  1 };
   enum { gje3Pad_19       =  2 };
   enum { gje3Pad_20       =  2 };
   enum { gje3Pad_21       =  0 };
   enum { gje3Pad_22       =  4 };
   enum { gje3Pad_23       =  2 };
   enum { gje3Pad_24       =  1 };
   enum { gje3Pad_25       =  4 };
   enum { gje3Pad_26       =  4 };
   enum { gje3Pad_27       =  3 };
   enum { gje3Pad_28       =  2 };
   enum { gje3Pad_29       =  0 };
   enum { gje3Pad_30       =  0 };
   enum { gje3Pad_31       =  0 };
   enum { gje3Pad_32       =  1 };
   enum { gje3Pad_33       =  2 };
   enum { gje3Pad_34       =  1 };
   enum { gje3Pad_35       =  4 };
   enum { gje3Pad_36       =  3 };
   enum { gje3Pad_37       =  1 };
   enum { gje3Pad_38       =  3 };
   enum { gje3Pad_39       =  2 };
   enum { gje3Pad_40       =  1 };
   enum { gje3Pad_41       =  2 };
   enum { gje3Pad_42       =  4 };
   enum { gje3Pad_43       =  2 };
   enum { gje3Pad_44       =  1 };
   enum { gje3Pad_45       =  0 };
   enum { gje3Pad_46       =  0 };
   enum { gje3Pad_47       =  0 };
   enum { gje3Pad_48       =  0 };
   enum { gje3Pad_49       =  0 };
   enum { gje3Pad_50       =  0 };
   enum { gje3Pad_51       =  0 };
   enum { gje3Pad_52       =  0 };
   enum { gje3Pad_53       =  0 };
   enum { gje3Pad_54       =  0 };
   enum { gje3Pad_55       =  0 };
   enum { gje3Pad_56       =  0 };
   enum { gje3Pad_57       =  0 };
   enum { gje3Pad_58       =  0 };
   enum { gje3Pad_59       =  0 };
   enum { gje3Pad_60       =  0 };
   enum { gje3Pad_61       =  0 };
   enum { gje3Pad_62       =  0 };
   enum { gje3Pad_63       =  0 };
   enum { gje3Pad_64       =  0 };
   enum { gje3Pad_65       =  0 };
   enum { gje3Pad_66       =  0 };
   enum { gje3Pad_67       =  0 };
   enum { gje3Pad_68       =  0 };
   enum { gje3Pad_69       =  0 };
   enum { gje3Pad_70       =  0 };
   enum { gje3Pad_71       =  0 };
   enum { gje3Pad_72       =  0 };
   enum { gje3Pad_73       =  0 };
   enum { gje3Pad_74       =  0 };
   enum { gje3Pad_75       =  0 };
   enum { gje3Pad_76       =  0 };
   enum { gje3Pad_77       =  0 };

   enum { gje3SrchThrd_00  = -1 };
   enum { gje3SrchThrd_01  = -1 };
   enum { gje3SrchThrd_02  =  1 };
   enum { gje3SrchThrd_03  =  2 };    
   enum { gje3SrchThrd_04  =  2 };
   enum { gje3SrchThrd_05  =  2 };
   enum { gje3SrchThrd_06  =  2 };
   enum { gje3SrchThrd_07  =  2 };
   enum { gje3SrchThrd_08  =  2 };
   enum { gje3SrchThrd_09  =  2 };
   enum { gje3SrchThrd_10  =  2 };
   enum { gje3SrchThrd_11  =  2 };
   enum { gje3SrchThrd_12  =  2 };
   enum { gje3SrchThrd_13  =  3 };
   enum { gje3SrchThrd_14  =  3 };
   enum { gje3SrchThrd_15  =  3 };
   enum { gje3SrchThrd_16  =  3 };
   enum { gje3SrchThrd_17  =  3 };
   enum { gje3SrchThrd_18  =  3 };
   enum { gje3SrchThrd_19  =  3 };
   enum { gje3SrchThrd_20  =  3 };
   enum { gje3SrchThrd_21  =  3 };
   enum { gje3SrchThrd_22  =  3 };
   enum { gje3SrchThrd_23  =  3 };
   enum { gje3SrchThrd_24  =  3 };
   enum { gje3SrchThrd_25  =  3 };
   enum { gje3SrchThrd_26  =  3 };
   enum { gje3SrchThrd_27  =  3 };
   enum { gje3SrchThrd_28  =  3 };
   enum { gje3SrchThrd_29  =  3 };
   enum { gje3SrchThrd_30  =  3 };
   enum { gje3SrchThrd_31  =  3 };
   enum { gje3SrchThrd_32  =  3 };
   enum { gje3SrchThrd_33  =  3 };
   enum { gje3SrchThrd_34  =  3 };
   enum { gje3SrchThrd_35  =  3 };
   enum { gje3SrchThrd_36  =  4 };
   enum { gje3SrchThrd_37  =  4 };
   enum { gje3SrchThrd_38  =  4 };
   enum { gje3SrchThrd_39  =  4 };
   enum { gje3SrchThrd_40  =  4 };
   enum { gje3SrchThrd_41  =  4 };
   enum { gje3SrchThrd_42  =  4 };
   enum { gje3SrchThrd_43  =  4 };
   enum { gje3SrchThrd_44  =  4 };
   enum { gje3SrchThrd_45  = -1 };
   enum { gje3SrchThrd_46  = -1 };
   enum { gje3SrchThrd_47  = -1 };
   enum { gje3SrchThrd_48  = -1 };
   enum { gje3SrchThrd_49  = -1 };
   enum { gje3SrchThrd_50  = -1 };
   enum { gje3SrchThrd_51  = -1 };
   enum { gje3SrchThrd_52  = -1 };
   enum { gje3SrchThrd_53  = -1 };
   enum { gje3SrchThrd_54  = -1 };
   enum { gje3SrchThrd_55  = -1 };
   enum { gje3SrchThrd_56  = -1 };
   enum { gje3SrchThrd_57  = -1 };
   enum { gje3SrchThrd_58  = -1 };
   enum { gje3SrchThrd_59  = -1 };
   enum { gje3SrchThrd_60  = -1 };
   enum { gje3SrchThrd_61  = -1 };
   enum { gje3SrchThrd_62  = -1 };
   enum { gje3SrchThrd_63  = -1 };
   enum { gje3SrchThrd_64  = -1 };
   enum { gje3SrchThrd_65  = -1 };
   enum { gje3SrchThrd_66  = -1 };
   enum { gje3SrchThrd_67  = -1 };
   enum { gje3SrchThrd_68  = -1 };
   enum { gje3SrchThrd_69  = -1 };
   enum { gje3SrchThrd_70  = -1 };
   enum { gje3SrchThrd_71  = -1 };
   enum { gje3SrchThrd_72  = -1 };
   enum { gje3SrchThrd_73  = -1 };
   enum { gje3SrchThrd_74  = -1 };
   enum { gje3SrchThrd_75  = -1 };
   enum { gje3SrchThrd_76  = -1 };
   enum { gje3SrchThrd_77  = -1 };
};

template<> class config<cuDoubleComplex,ARCH_SM13> {
public:
   enum { gje3MinDim       =  2 };
   enum { gje3MaxDim       = 31 };
   enum { gje3MinBlks    =    1 };
   enum { gje3MaxThrds   =  640 }; /* sm_13, 25 registers per thread */

   enum { gje3DimX_00      = -1 };
   enum { gje3DimX_01      = -1 };
   enum { gje3DimX_02      =  2 };
   enum { gje3DimX_03      =  3 };
   enum { gje3DimX_04      =  4 };
   enum { gje3DimX_05      =  3 };
   enum { gje3DimX_06      =  2 };
   enum { gje3DimX_07      =  2 };
   enum { gje3DimX_08      =  2 };
   enum { gje3DimX_09      =  3 };
   enum { gje3DimX_10      =  3 };
   enum { gje3DimX_11      =  3 };
   enum { gje3DimX_12      =  4 };
   enum { gje3DimX_13      =  3 };
   enum { gje3DimX_14      =  3 };
   enum { gje3DimX_15      =  3 };
   enum { gje3DimX_16      =  4 };
   enum { gje3DimX_17      =  4 };
   enum { gje3DimX_18      =  4 };
   enum { gje3DimX_19      =  4 };
   enum { gje3DimX_20      =  4 };
   enum { gje3DimX_21      =  5 };
   enum { gje3DimX_22      =  5 };
   enum { gje3DimX_23      =  6 };
   enum { gje3DimX_24      =  6 };
   enum { gje3DimX_25      =  5 };
   enum { gje3DimX_26      =  6 };
   enum { gje3DimX_27      =  7 };
   enum { gje3DimX_28      =  4 };
   enum { gje3DimX_29      =  6 };
   enum { gje3DimX_30      =  8 };
   enum { gje3DimX_31      =  4 };
   enum { gje3DimX_32      = -1 };
   enum { gje3DimX_33      = -1 };
   enum { gje3DimX_34      = -1 };
   enum { gje3DimX_35      = -1 };
   enum { gje3DimX_36      = -1 };
   enum { gje3DimX_37      = -1 };
   enum { gje3DimX_38      = -1 };
   enum { gje3DimX_39      = -1 };
   enum { gje3DimX_40      = -1 };
   enum { gje3DimX_41      = -1 };
   enum { gje3DimX_42      = -1 };
   enum { gje3DimX_43      = -1 };
   enum { gje3DimX_44      = -1 };
   enum { gje3DimX_45      = -1 };
   enum { gje3DimX_46      = -1 };
   enum { gje3DimX_47      = -1 };
   enum { gje3DimX_48      = -1 };
   enum { gje3DimX_49      = -1 };
   enum { gje3DimX_50      = -1 };
   enum { gje3DimX_51      = -1 };
   enum { gje3DimX_52      = -1 };
   enum { gje3DimX_53      = -1 };
   enum { gje3DimX_54      = -1 };
   enum { gje3DimX_55      = -1 };
   enum { gje3DimX_56      = -1 };
   enum { gje3DimX_57      = -1 };
   enum { gje3DimX_58      = -1 };
   enum { gje3DimX_59      = -1 };
   enum { gje3DimX_60      = -1 };
   enum { gje3DimX_61      = -1 };
   enum { gje3DimX_62      = -1 };
   enum { gje3DimX_63      = -1 };
   enum { gje3DimX_64      = -1 };
   enum { gje3DimX_65      = -1 };
   enum { gje3DimX_66      = -1 };
   enum { gje3DimX_67      = -1 };
   enum { gje3DimX_68      = -1 };
   enum { gje3DimX_69      = -1 };
   enum { gje3DimX_70      = -1 };
   enum { gje3DimX_71      = -1 };
   enum { gje3DimX_72      = -1 };
   enum { gje3DimX_73      = -1 };
   enum { gje3DimX_74      = -1 };
   enum { gje3DimX_75      = -1 };
   enum { gje3DimX_76      = -1 };
   enum { gje3DimX_77      = -1 };

   enum { gje3Pad_00       =  0 };
   enum { gje3Pad_01       =  0 };
   enum { gje3Pad_02       =  0 };
   enum { gje3Pad_03       =  0 };
   enum { gje3Pad_04       =  1 };
   enum { gje3Pad_05       =  0 };
   enum { gje3Pad_06       =  1 };
   enum { gje3Pad_07       =  0 };
   enum { gje3Pad_08       =  1 };
   enum { gje3Pad_09       =  2 };
   enum { gje3Pad_10       =  1 };
   enum { gje3Pad_11       =  0 };
   enum { gje3Pad_12       =  1 };
   enum { gje3Pad_13       =  0 };
   enum { gje3Pad_14       =  1 };
   enum { gje3Pad_15       =  0 };
   enum { gje3Pad_16       =  1 };
   enum { gje3Pad_17       =  0 };
   enum { gje3Pad_18       =  1 };
   enum { gje3Pad_19       =  0 };
   enum { gje3Pad_20       =  1 };
   enum { gje3Pad_21       =  0 };
   enum { gje3Pad_22       =  0 };
   enum { gje3Pad_23       =  0 };
   enum { gje3Pad_24       =  1 };
   enum { gje3Pad_25       =  0 };
   enum { gje3Pad_26       =  0 };
   enum { gje3Pad_27       =  0 };
   enum { gje3Pad_28       =  1 };
   enum { gje3Pad_29       =  0 };
   enum { gje3Pad_30       =  0 };
   enum { gje3Pad_31       =  0 };
   enum { gje3Pad_32       =  0 };
   enum { gje3Pad_33       =  0 };
   enum { gje3Pad_34       =  0 };
   enum { gje3Pad_35       =  0 };
   enum { gje3Pad_36       =  0 };
   enum { gje3Pad_37       =  0 };
   enum { gje3Pad_38       =  0 };
   enum { gje3Pad_39       =  0 };
   enum { gje3Pad_40       =  0 };
   enum { gje3Pad_41       =  0 };
   enum { gje3Pad_42       =  0 };
   enum { gje3Pad_43       =  0 };
   enum { gje3Pad_44       =  0 };
   enum { gje3Pad_45       =  0 };
   enum { gje3Pad_46       =  0 };
   enum { gje3Pad_47       =  0 };
   enum { gje3Pad_48       =  0 };
   enum { gje3Pad_49       =  0 };
   enum { gje3Pad_50       =  0 };
   enum { gje3Pad_51       =  0 };
   enum { gje3Pad_52       =  0 };
   enum { gje3Pad_53       =  0 };
   enum { gje3Pad_54       =  0 };
   enum { gje3Pad_55       =  0 };
   enum { gje3Pad_56       =  0 };
   enum { gje3Pad_57       =  0 };
   enum { gje3Pad_58       =  0 };
   enum { gje3Pad_59       =  0 };
   enum { gje3Pad_60       =  0 };
   enum { gje3Pad_61       =  0 };
   enum { gje3Pad_62       =  0 };
   enum { gje3Pad_63       =  0 };
   enum { gje3Pad_64       =  0 };
   enum { gje3Pad_65       =  0 };
   enum { gje3Pad_66       =  0 };
   enum { gje3Pad_67       =  0 };
   enum { gje3Pad_68       =  0 };
   enum { gje3Pad_69       =  0 };
   enum { gje3Pad_70       =  0 };
   enum { gje3Pad_71       =  0 };
   enum { gje3Pad_72       =  0 };
   enum { gje3Pad_73       =  0 };
   enum { gje3Pad_74       =  0 };
   enum { gje3Pad_75       =  0 };
   enum { gje3Pad_76       =  0 };
   enum { gje3Pad_77       =  0 };

   enum { gje3SrchThrd_00  = -1 };
   enum { gje3SrchThrd_01  = -1 };
   enum { gje3SrchThrd_02  =  1 };
   enum { gje3SrchThrd_03  =  2 };
   enum { gje3SrchThrd_04  =  2 };
   enum { gje3SrchThrd_05  =  2 };
   enum { gje3SrchThrd_06  =  2 };
   enum { gje3SrchThrd_07  =  2 };
   enum { gje3SrchThrd_08  =  3 };
   enum { gje3SrchThrd_09  =  3 };
   enum { gje3SrchThrd_10  =  3 };
   enum { gje3SrchThrd_11  =  3 };
   enum { gje3SrchThrd_12  =  3 };
   enum { gje3SrchThrd_13  =  3 };
   enum { gje3SrchThrd_14  =  3 };
   enum { gje3SrchThrd_15  =  3 };
   enum { gje3SrchThrd_16  =  3 };
   enum { gje3SrchThrd_17  =  3 };
   enum { gje3SrchThrd_18  =  3 };
   enum { gje3SrchThrd_19  =  3 };
   enum { gje3SrchThrd_20  =  4 };
   enum { gje3SrchThrd_21  =  4 };
   enum { gje3SrchThrd_22  =  4 };
   enum { gje3SrchThrd_23  =  4 };
   enum { gje3SrchThrd_24  =  4 };
   enum { gje3SrchThrd_25  =  4 };
   enum { gje3SrchThrd_26  =  4 };
   enum { gje3SrchThrd_27  =  4 };
   enum { gje3SrchThrd_28  =  4 };
   enum { gje3SrchThrd_29  =  4 };
   enum { gje3SrchThrd_30  =  4 };
   enum { gje3SrchThrd_31  =  4 };
   enum { gje3SrchThrd_32  = -1 };
   enum { gje3SrchThrd_33  = -1 };
   enum { gje3SrchThrd_34  = -1 };
   enum { gje3SrchThrd_35  = -1 };
   enum { gje3SrchThrd_36  = -1 };
   enum { gje3SrchThrd_37  = -1 };
   enum { gje3SrchThrd_38  = -1 };
   enum { gje3SrchThrd_39  = -1 };
   enum { gje3SrchThrd_40  = -1 };
   enum { gje3SrchThrd_41  = -1 };
   enum { gje3SrchThrd_42  = -1 };
   enum { gje3SrchThrd_43  = -1 };
   enum { gje3SrchThrd_44  = -1 };
   enum { gje3SrchThrd_45  = -1 };
   enum { gje3SrchThrd_46  = -1 };
   enum { gje3SrchThrd_47  = -1 };
   enum { gje3SrchThrd_48  = -1 };
   enum { gje3SrchThrd_49  = -1 };
   enum { gje3SrchThrd_50  = -1 };
   enum { gje3SrchThrd_51  = -1 };
   enum { gje3SrchThrd_52  = -1 };
   enum { gje3SrchThrd_53  = -1 };
   enum { gje3SrchThrd_54  = -1 };
   enum { gje3SrchThrd_55  = -1 };
   enum { gje3SrchThrd_56  = -1 };
   enum { gje3SrchThrd_57  = -1 };
   enum { gje3SrchThrd_58  = -1 };
   enum { gje3SrchThrd_59  = -1 };
   enum { gje3SrchThrd_60  = -1 };
   enum { gje3SrchThrd_61  = -1 };
   enum { gje3SrchThrd_62  = -1 };
   enum { gje3SrchThrd_63  = -1 };
   enum { gje3SrchThrd_64  = -1 };
   enum { gje3SrchThrd_65  = -1 };
   enum { gje3SrchThrd_66  = -1 };
   enum { gje3SrchThrd_67  = -1 };
   enum { gje3SrchThrd_68  = -1 };
   enum { gje3SrchThrd_69  = -1 };
   enum { gje3SrchThrd_70  = -1 };
   enum { gje3SrchThrd_71  = -1 };
   enum { gje3SrchThrd_72  = -1 };
   enum { gje3SrchThrd_73  = -1 };
   enum { gje3SrchThrd_74  = -1 };
   enum { gje3SrchThrd_75  = -1 };
   enum { gje3SrchThrd_76  = -1 };
   enum { gje3SrchThrd_77  = -1 };
};

// cuComplex and float are just copies of the double config (however float should be something different)
template<> class config<cuComplex,ARCH_SM13> {
public:
   enum { gje3MinDim       =  2 };
   enum { gje3MaxDim       = 44 };
   enum { gje3MinBlks      =  1 };
   enum { gje3MaxThrds     =768 }; /* sm_13, 21 registers per thread */

   enum { gje3DimX_00      = -1 };
   enum { gje3DimX_01      = -1 };
   enum { gje3DimX_02      =  2 };
   enum { gje3DimX_03      =  3 };
   enum { gje3DimX_04      =  4 };
   enum { gje3DimX_05      =  3 };
   enum { gje3DimX_06      =  2 };
   enum { gje3DimX_07      =  2 };
   enum { gje3DimX_08      =  2 };
   enum { gje3DimX_09      =  3 };
   enum { gje3DimX_10      =  3 };
   enum { gje3DimX_11      =  2 };
   enum { gje3DimX_12      =  4 };
   enum { gje3DimX_13      =  2 };
   enum { gje3DimX_14      =  2 };
   enum { gje3DimX_15      =  2 };
   enum { gje3DimX_16      =  2 };
   enum { gje3DimX_17      =  2 };
   enum { gje3DimX_18      =  2 };
   enum { gje3DimX_19      =  3 };
   enum { gje3DimX_20      =  4 };
   enum { gje3DimX_21      =  3 };
   enum { gje3DimX_22      =  4 };
   enum { gje3DimX_23      =  2 };
   enum { gje3DimX_24      =  2 };
   enum { gje3DimX_25      =  5 };
   enum { gje3DimX_26      =  4 };
   enum { gje3DimX_27      =  4 };
   enum { gje3DimX_28      =  4 };
   enum { gje3DimX_29      =  5 };
   enum { gje3DimX_30      =  4 };
   enum { gje3DimX_31      =  2 };
   enum { gje3DimX_32      =  8 };
   enum { gje3DimX_33      =  7 };
   enum { gje3DimX_34      =  7 };
   enum { gje3DimX_35      =  7 };
   enum { gje3DimX_36      =  8 };
   enum { gje3DimX_37      =  8 };
   enum { gje3DimX_38      =  8 };
   enum { gje3DimX_39      =  8 };
   enum { gje3DimX_40      =  8 };
   enum { gje3DimX_41      =  7 };
   enum { gje3DimX_42      =  6 };
   enum { gje3DimX_43      =  8 };
   enum { gje3DimX_44      =  8 };
   enum { gje3DimX_45      = -1 };
   enum { gje3DimX_46      = -1 };
   enum { gje3DimX_47      = -1 };
   enum { gje3DimX_48      = -1 };
   enum { gje3DimX_49      = -1 };
   enum { gje3DimX_50      = -1 };
   enum { gje3DimX_51      = -1 };
   enum { gje3DimX_52      = -1 };
   enum { gje3DimX_53      = -1 };
   enum { gje3DimX_54      = -1 };
   enum { gje3DimX_55      = -1 };
   enum { gje3DimX_56      = -1 };
   enum { gje3DimX_57      = -1 };
   enum { gje3DimX_58      = -1 };
   enum { gje3DimX_59      = -1 };
   enum { gje3DimX_60      = -1 };
   enum { gje3DimX_61      = -1 };
   enum { gje3DimX_62      = -1 };
   enum { gje3DimX_63      = -1 };
   enum { gje3DimX_64      = -1 };
   enum { gje3DimX_65      = -1 };
   enum { gje3DimX_66      = -1 };
   enum { gje3DimX_67      = -1 };
   enum { gje3DimX_68      = -1 };
   enum { gje3DimX_69      = -1 };
   enum { gje3DimX_70      = -1 };
   enum { gje3DimX_71      = -1 };
   enum { gje3DimX_72      = -1 };
   enum { gje3DimX_73      = -1 };
   enum { gje3DimX_74      = -1 };
   enum { gje3DimX_75      = -1 };
   enum { gje3DimX_76      = -1 };
   enum { gje3DimX_77      = -1 };

   enum { gje3Pad_00       =  0 };
   enum { gje3Pad_01       =  0 };
   enum { gje3Pad_02       =  0 };
   enum { gje3Pad_03       =  0 };
   enum { gje3Pad_04       =  2 };
   enum { gje3Pad_05       =  0 };
   enum { gje3Pad_06       =  1 };
   enum { gje3Pad_07       =  4 };
   enum { gje3Pad_08       =  3 };
   enum { gje3Pad_09       =  2 };
   enum { gje3Pad_10       =  1 };
   enum { gje3Pad_11       =  2 };
   enum { gje3Pad_12       =  2 };
   enum { gje3Pad_13       =  2 };
   enum { gje3Pad_14       =  1 };
   enum { gje3Pad_15       =  0 };
   enum { gje3Pad_16       =  1 };
   enum { gje3Pad_17       =  0 };
   enum { gje3Pad_18       =  1 };
   enum { gje3Pad_19       =  2 };
   enum { gje3Pad_20       =  2 };
   enum { gje3Pad_21       =  0 };
   enum { gje3Pad_22       =  4 };
   enum { gje3Pad_23       =  2 };
   enum { gje3Pad_24       =  1 };
   enum { gje3Pad_25       =  4 };
   enum { gje3Pad_26       =  4 };
   enum { gje3Pad_27       =  3 };
   enum { gje3Pad_28       =  2 };
   enum { gje3Pad_29       =  0 };
   enum { gje3Pad_30       =  0 };
   enum { gje3Pad_31       =  0 };
   enum { gje3Pad_32       =  1 };
   enum { gje3Pad_33       =  2 };
   enum { gje3Pad_34       =  1 };
   enum { gje3Pad_35       =  4 };
   enum { gje3Pad_36       =  3 };
   enum { gje3Pad_37       =  1 };
   enum { gje3Pad_38       =  3 };
   enum { gje3Pad_39       =  2 };
   enum { gje3Pad_40       =  1 };
   enum { gje3Pad_41       =  2 };
   enum { gje3Pad_42       =  4 };
   enum { gje3Pad_43       =  2 };
   enum { gje3Pad_44       =  1 };
   enum { gje3Pad_45       =  0 };
   enum { gje3Pad_46       =  0 };
   enum { gje3Pad_47       =  0 };
   enum { gje3Pad_48       =  0 };
   enum { gje3Pad_49       =  0 };
   enum { gje3Pad_50       =  0 };
   enum { gje3Pad_51       =  0 };
   enum { gje3Pad_52       =  0 };
   enum { gje3Pad_53       =  0 };
   enum { gje3Pad_54       =  0 };
   enum { gje3Pad_55       =  0 };
   enum { gje3Pad_56       =  0 };
   enum { gje3Pad_57       =  0 };
   enum { gje3Pad_58       =  0 };
   enum { gje3Pad_59       =  0 };
   enum { gje3Pad_60       =  0 };
   enum { gje3Pad_61       =  0 };
   enum { gje3Pad_62       =  0 };
   enum { gje3Pad_63       =  0 };
   enum { gje3Pad_64       =  0 };
   enum { gje3Pad_65       =  0 };
   enum { gje3Pad_66       =  0 };
   enum { gje3Pad_67       =  0 };
   enum { gje3Pad_68       =  0 };
   enum { gje3Pad_69       =  0 };
   enum { gje3Pad_70       =  0 };
   enum { gje3Pad_71       =  0 };
   enum { gje3Pad_72       =  0 };
   enum { gje3Pad_73       =  0 };
   enum { gje3Pad_74       =  0 };
   enum { gje3Pad_75       =  0 };
   enum { gje3Pad_76       =  0 };
   enum { gje3Pad_77       =  0 };

   enum { gje3SrchThrd_00  = -1 };
   enum { gje3SrchThrd_01  = -1 };
   enum { gje3SrchThrd_02  =  1 };
   enum { gje3SrchThrd_03  =  2 };    
   enum { gje3SrchThrd_04  =  2 };
   enum { gje3SrchThrd_05  =  2 };
   enum { gje3SrchThrd_06  =  2 };
   enum { gje3SrchThrd_07  =  2 };
   enum { gje3SrchThrd_08  =  2 };
   enum { gje3SrchThrd_09  =  2 };
   enum { gje3SrchThrd_10  =  2 };
   enum { gje3SrchThrd_11  =  2 };
   enum { gje3SrchThrd_12  =  2 };
   enum { gje3SrchThrd_13  =  3 };
   enum { gje3SrchThrd_14  =  3 };
   enum { gje3SrchThrd_15  =  3 };
   enum { gje3SrchThrd_16  =  3 };
   enum { gje3SrchThrd_17  =  3 };
   enum { gje3SrchThrd_18  =  3 };
   enum { gje3SrchThrd_19  =  3 };
   enum { gje3SrchThrd_20  =  3 };
   enum { gje3SrchThrd_21  =  3 };
   enum { gje3SrchThrd_22  =  3 };
   enum { gje3SrchThrd_23  =  3 };
   enum { gje3SrchThrd_24  =  3 };
   enum { gje3SrchThrd_25  =  3 };
   enum { gje3SrchThrd_26  =  3 };
   enum { gje3SrchThrd_27  =  3 };
   enum { gje3SrchThrd_28  =  3 };
   enum { gje3SrchThrd_29  =  3 };
   enum { gje3SrchThrd_30  =  3 };
   enum { gje3SrchThrd_31  =  3 };
   enum { gje3SrchThrd_32  =  3 };
   enum { gje3SrchThrd_33  =  3 };
   enum { gje3SrchThrd_34  =  3 };
   enum { gje3SrchThrd_35  =  3 };
   enum { gje3SrchThrd_36  =  4 };
   enum { gje3SrchThrd_37  =  4 };
   enum { gje3SrchThrd_38  =  4 };
   enum { gje3SrchThrd_39  =  4 };
   enum { gje3SrchThrd_40  =  4 };
   enum { gje3SrchThrd_41  =  4 };
   enum { gje3SrchThrd_42  =  4 };
   enum { gje3SrchThrd_43  =  4 };
   enum { gje3SrchThrd_44  =  4 };
   enum { gje3SrchThrd_45  = -1 };
   enum { gje3SrchThrd_46  = -1 };
   enum { gje3SrchThrd_47  = -1 };
   enum { gje3SrchThrd_48  = -1 };
   enum { gje3SrchThrd_49  = -1 };
   enum { gje3SrchThrd_50  = -1 };
   enum { gje3SrchThrd_51  = -1 };
   enum { gje3SrchThrd_52  = -1 };
   enum { gje3SrchThrd_53  = -1 };
   enum { gje3SrchThrd_54  = -1 };
   enum { gje3SrchThrd_55  = -1 };
   enum { gje3SrchThrd_56  = -1 };
   enum { gje3SrchThrd_57  = -1 };
   enum { gje3SrchThrd_58  = -1 };
   enum { gje3SrchThrd_59  = -1 };
   enum { gje3SrchThrd_60  = -1 };
   enum { gje3SrchThrd_61  = -1 };
   enum { gje3SrchThrd_62  = -1 };
   enum { gje3SrchThrd_63  = -1 };
   enum { gje3SrchThrd_64  = -1 };
   enum { gje3SrchThrd_65  = -1 };
   enum { gje3SrchThrd_66  = -1 };
   enum { gje3SrchThrd_67  = -1 };
   enum { gje3SrchThrd_68  = -1 };
   enum { gje3SrchThrd_69  = -1 };
   enum { gje3SrchThrd_70  = -1 };
   enum { gje3SrchThrd_71  = -1 };
   enum { gje3SrchThrd_72  = -1 };
   enum { gje3SrchThrd_73  = -1 };
   enum { gje3SrchThrd_74  = -1 };
   enum { gje3SrchThrd_75  = -1 };
   enum { gje3SrchThrd_76  = -1 };
   enum { gje3SrchThrd_77  = -1 };
};

template<> class config<float,ARCH_SM13> {
public:
   enum { gje3MinDim       =  2 };
   enum { gje3MaxDim       = 44 };
   enum { gje3MinBlks      =  1 };
   enum { gje3MaxThrds     =768 }; /* sm_13, 21 registers per thread */

   enum { gje3DimX_00      = -1 };
   enum { gje3DimX_01      = -1 };
   enum { gje3DimX_02      =  2 };
   enum { gje3DimX_03      =  3 };
   enum { gje3DimX_04      =  4 };
   enum { gje3DimX_05      =  3 };
   enum { gje3DimX_06      =  2 };
   enum { gje3DimX_07      =  2 };
   enum { gje3DimX_08      =  2 };
   enum { gje3DimX_09      =  3 };
   enum { gje3DimX_10      =  3 };
   enum { gje3DimX_11      =  2 };
   enum { gje3DimX_12      =  4 };
   enum { gje3DimX_13      =  2 };
   enum { gje3DimX_14      =  2 };
   enum { gje3DimX_15      =  2 };
   enum { gje3DimX_16      =  2 };
   enum { gje3DimX_17      =  2 };
   enum { gje3DimX_18      =  2 };
   enum { gje3DimX_19      =  3 };
   enum { gje3DimX_20      =  4 };
   enum { gje3DimX_21      =  3 };
   enum { gje3DimX_22      =  4 };
   enum { gje3DimX_23      =  2 };
   enum { gje3DimX_24      =  2 };
   enum { gje3DimX_25      =  5 };
   enum { gje3DimX_26      =  4 };
   enum { gje3DimX_27      =  4 };
   enum { gje3DimX_28      =  4 };
   enum { gje3DimX_29      =  5 };
   enum { gje3DimX_30      =  4 };
   enum { gje3DimX_31      =  2 };
   enum { gje3DimX_32      =  8 };
   enum { gje3DimX_33      =  7 };
   enum { gje3DimX_34      =  7 };
   enum { gje3DimX_35      =  7 };
   enum { gje3DimX_36      =  8 };
   enum { gje3DimX_37      =  8 };
   enum { gje3DimX_38      =  8 };
   enum { gje3DimX_39      =  8 };
   enum { gje3DimX_40      =  8 };
   enum { gje3DimX_41      =  7 };
   enum { gje3DimX_42      =  6 };
   enum { gje3DimX_43      =  8 };
   enum { gje3DimX_44      =  8 };
   enum { gje3DimX_45      = -1 };
   enum { gje3DimX_46      = -1 };
   enum { gje3DimX_47      = -1 };
   enum { gje3DimX_48      = -1 };
   enum { gje3DimX_49      = -1 };
   enum { gje3DimX_50      = -1 };
   enum { gje3DimX_51      = -1 };
   enum { gje3DimX_52      = -1 };
   enum { gje3DimX_53      = -1 };
   enum { gje3DimX_54      = -1 };
   enum { gje3DimX_55      = -1 };
   enum { gje3DimX_56      = -1 };
   enum { gje3DimX_57      = -1 };
   enum { gje3DimX_58      = -1 };
   enum { gje3DimX_59      = -1 };
   enum { gje3DimX_60      = -1 };
   enum { gje3DimX_61      = -1 };
   enum { gje3DimX_62      = -1 };
   enum { gje3DimX_63      = -1 };
   enum { gje3DimX_64      = -1 };
   enum { gje3DimX_65      = -1 };
   enum { gje3DimX_66      = -1 };
   enum { gje3DimX_67      = -1 };
   enum { gje3DimX_68      = -1 };
   enum { gje3DimX_69      = -1 };
   enum { gje3DimX_70      = -1 };
   enum { gje3DimX_71      = -1 };
   enum { gje3DimX_72      = -1 };
   enum { gje3DimX_73      = -1 };
   enum { gje3DimX_74      = -1 };
   enum { gje3DimX_75      = -1 };
   enum { gje3DimX_76      = -1 };
   enum { gje3DimX_77      = -1 };

   enum { gje3Pad_00       =  0 };
   enum { gje3Pad_01       =  0 };
   enum { gje3Pad_02       =  0 };
   enum { gje3Pad_03       =  0 };
   enum { gje3Pad_04       =  2 };
   enum { gje3Pad_05       =  0 };
   enum { gje3Pad_06       =  1 };
   enum { gje3Pad_07       =  4 };
   enum { gje3Pad_08       =  3 };
   enum { gje3Pad_09       =  2 };
   enum { gje3Pad_10       =  1 };
   enum { gje3Pad_11       =  2 };
   enum { gje3Pad_12       =  2 };
   enum { gje3Pad_13       =  2 };
   enum { gje3Pad_14       =  1 };
   enum { gje3Pad_15       =  0 };
   enum { gje3Pad_16       =  1 };
   enum { gje3Pad_17       =  0 };
   enum { gje3Pad_18       =  1 };
   enum { gje3Pad_19       =  2 };
   enum { gje3Pad_20       =  2 };
   enum { gje3Pad_21       =  0 };
   enum { gje3Pad_22       =  4 };
   enum { gje3Pad_23       =  2 };
   enum { gje3Pad_24       =  1 };
   enum { gje3Pad_25       =  4 };
   enum { gje3Pad_26       =  4 };
   enum { gje3Pad_27       =  3 };
   enum { gje3Pad_28       =  2 };
   enum { gje3Pad_29       =  0 };
   enum { gje3Pad_30       =  0 };
   enum { gje3Pad_31       =  0 };
   enum { gje3Pad_32       =  1 };
   enum { gje3Pad_33       =  2 };
   enum { gje3Pad_34       =  1 };
   enum { gje3Pad_35       =  4 };
   enum { gje3Pad_36       =  3 };
   enum { gje3Pad_37       =  1 };
   enum { gje3Pad_38       =  3 };
   enum { gje3Pad_39       =  2 };
   enum { gje3Pad_40       =  1 };
   enum { gje3Pad_41       =  2 };
   enum { gje3Pad_42       =  4 };
   enum { gje3Pad_43       =  2 };
   enum { gje3Pad_44       =  1 };
   enum { gje3Pad_45       =  0 };
   enum { gje3Pad_46       =  0 };
   enum { gje3Pad_47       =  0 };
   enum { gje3Pad_48       =  0 };
   enum { gje3Pad_49       =  0 };
   enum { gje3Pad_50       =  0 };
   enum { gje3Pad_51       =  0 };
   enum { gje3Pad_52       =  0 };
   enum { gje3Pad_53       =  0 };
   enum { gje3Pad_54       =  0 };
   enum { gje3Pad_55       =  0 };
   enum { gje3Pad_56       =  0 };
   enum { gje3Pad_57       =  0 };
   enum { gje3Pad_58       =  0 };
   enum { gje3Pad_59       =  0 };
   enum { gje3Pad_60       =  0 };
   enum { gje3Pad_61       =  0 };
   enum { gje3Pad_62       =  0 };
   enum { gje3Pad_63       =  0 };
   enum { gje3Pad_64       =  0 };
   enum { gje3Pad_65       =  0 };
   enum { gje3Pad_66       =  0 };
   enum { gje3Pad_67       =  0 };
   enum { gje3Pad_68       =  0 };
   enum { gje3Pad_69       =  0 };
   enum { gje3Pad_70       =  0 };
   enum { gje3Pad_71       =  0 };
   enum { gje3Pad_72       =  0 };
   enum { gje3Pad_73       =  0 };
   enum { gje3Pad_74       =  0 };
   enum { gje3Pad_75       =  0 };
   enum { gje3Pad_76       =  0 };
   enum { gje3Pad_77       =  0 };

   enum { gje3SrchThrd_00  = -1 };
   enum { gje3SrchThrd_01  = -1 };
   enum { gje3SrchThrd_02  =  1 };
   enum { gje3SrchThrd_03  =  2 };    
   enum { gje3SrchThrd_04  =  2 };
   enum { gje3SrchThrd_05  =  2 };
   enum { gje3SrchThrd_06  =  2 };
   enum { gje3SrchThrd_07  =  2 };
   enum { gje3SrchThrd_08  =  2 };
   enum { gje3SrchThrd_09  =  2 };
   enum { gje3SrchThrd_10  =  2 };
   enum { gje3SrchThrd_11  =  2 };
   enum { gje3SrchThrd_12  =  2 };
   enum { gje3SrchThrd_13  =  3 };
   enum { gje3SrchThrd_14  =  3 };
   enum { gje3SrchThrd_15  =  3 };
   enum { gje3SrchThrd_16  =  3 };
   enum { gje3SrchThrd_17  =  3 };
   enum { gje3SrchThrd_18  =  3 };
   enum { gje3SrchThrd_19  =  3 };
   enum { gje3SrchThrd_20  =  3 };
   enum { gje3SrchThrd_21  =  3 };
   enum { gje3SrchThrd_22  =  3 };
   enum { gje3SrchThrd_23  =  3 };
   enum { gje3SrchThrd_24  =  3 };
   enum { gje3SrchThrd_25  =  3 };
   enum { gje3SrchThrd_26  =  3 };
   enum { gje3SrchThrd_27  =  3 };
   enum { gje3SrchThrd_28  =  3 };
   enum { gje3SrchThrd_29  =  3 };
   enum { gje3SrchThrd_30  =  3 };
   enum { gje3SrchThrd_31  =  3 };
   enum { gje3SrchThrd_32  =  3 };
   enum { gje3SrchThrd_33  =  3 };
   enum { gje3SrchThrd_34  =  3 };
   enum { gje3SrchThrd_35  =  3 };
   enum { gje3SrchThrd_36  =  4 };
   enum { gje3SrchThrd_37  =  4 };
   enum { gje3SrchThrd_38  =  4 };
   enum { gje3SrchThrd_39  =  4 };
   enum { gje3SrchThrd_40  =  4 };
   enum { gje3SrchThrd_41  =  4 };
   enum { gje3SrchThrd_42  =  4 };
   enum { gje3SrchThrd_43  =  4 };
   enum { gje3SrchThrd_44  =  4 };
   enum { gje3SrchThrd_45  = -1 };
   enum { gje3SrchThrd_46  = -1 };
   enum { gje3SrchThrd_47  = -1 };
   enum { gje3SrchThrd_48  = -1 };
   enum { gje3SrchThrd_49  = -1 };
   enum { gje3SrchThrd_50  = -1 };
   enum { gje3SrchThrd_51  = -1 };
   enum { gje3SrchThrd_52  = -1 };
   enum { gje3SrchThrd_53  = -1 };
   enum { gje3SrchThrd_54  = -1 };
   enum { gje3SrchThrd_55  = -1 };
   enum { gje3SrchThrd_56  = -1 };
   enum { gje3SrchThrd_57  = -1 };
   enum { gje3SrchThrd_58  = -1 };
   enum { gje3SrchThrd_59  = -1 };
   enum { gje3SrchThrd_60  = -1 };
   enum { gje3SrchThrd_61  = -1 };
   enum { gje3SrchThrd_62  = -1 };
   enum { gje3SrchThrd_63  = -1 };
   enum { gje3SrchThrd_64  = -1 };
   enum { gje3SrchThrd_65  = -1 };
   enum { gje3SrchThrd_66  = -1 };
   enum { gje3SrchThrd_67  = -1 };
   enum { gje3SrchThrd_68  = -1 };
   enum { gje3SrchThrd_69  = -1 };
   enum { gje3SrchThrd_70  = -1 };
   enum { gje3SrchThrd_71  = -1 };
   enum { gje3SrchThrd_72  = -1 };
   enum { gje3SrchThrd_73  = -1 };
   enum { gje3SrchThrd_74  = -1 };
   enum { gje3SrchThrd_75  = -1 };
   enum { gje3SrchThrd_76  = -1 };
   enum { gje3SrchThrd_77  = -1 };
};

/* column-major */
#define As(row,col)   As[(N+ofs)*(col)+(row)]
#define AsInv(row,col)   AsInv[(N+ofs)*(col)+(row)]

extern __shared__ double2 shmem[];

template<typename T, int pad, int pivot_thrds, int arch>
__global__ void
__launch_bounds__ (config<T,arch>::gje3MaxThrds, config<T,arch>::gje3MinBlks)
matinv_gje3 (const T *A, T *Ainv, int N, int batch)
{
   T *As = (T*)shmem;
   double *Val= (double*)(As + (N+pad) * N);
   int *Loc = (int*)(Val + pivot_thrds);
   int *icol = (int*)(Loc + pivot_thrds);
   int *perm = (int*)(icol + N);
   T diagRcp;
   const int ofs = pad;
   const int tx = threadIdx.x;
   const int ty = threadIdx.y;
   const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;

   if (blkNum >= batch) return;

   A    += blkNum * N * N;
   Ainv += blkNum * N * N;

   /* Load matrix and into shared memory */
   for (int i = tx; i < N; i += blockDim.x) {
      As(i,ty) = A[ty * N + i];
   }
   /* initialize row permutation vector */
   if (tx == 0) perm[ty] = ty;

   int j = 0;
   do {
      /* Look for pivot */
      __syncthreads();
      if ((tx == 0) && (ty < pivot_thrds)) {
         double val0 = absOp (As(j,j));
         int loc0 = j;
         int i = j + 1 + ty;
         T *dp = &As(i,j);
         const int incr = &As(pivot_thrds,0)-&As(0,0);
         while (i < N) {
            double vali = absOp (*dp);
            if (val0 < vali) {
               val0 = vali;
               loc0 = i;
            }
            dp += incr;
            i  += pivot_thrds;
         }
         Loc[ty] = loc0;
         if (pivot_thrds > 1) Val[ty] = val0;
      }

      /* Swap current row with pivot */
      __syncthreads();
      if (tx == 0) {
         T tmp;
         int it;
         int Pl = Loc[0];
         if (pivot_thrds > 1) {
            double val = Val[0];
            int i = 1;
            for (; i < (pivot_thrds-1); i++) {
               if (Val[i] > val) { 
                  Pl = Loc[i]; 
                  val = Val[i]; 
               }
            }
            if (Val[i] > val) { 
               Pl = Loc[i]; 
            }
         }
         tmp = As(Pl,ty);
         As(Pl,ty) = As(j,ty);
         As(j,ty) = tmp;
         /* update permutation vector based on row swap */
         if (ty == j) {
            it = perm[Pl];
            perm[Pl] = perm[j];
            perm[j] = it;
         }
      }

      /* scale current row, except current column */
      __syncthreads();
      diagRcp = rcpOp (As(j,j));
      if ((tx == 0) && !(ty == j)) {
         As(j,ty) = mulOp (As(j,ty), diagRcp);
      }

      /* update above and below current row, except current column */
      __syncthreads();
      for (int i = tx; i < N; i += blockDim.x) {            
         if ((i != j) && !(ty == j)) {
            As(i,ty) = fmnaOp (As(i,j), As(j,ty), As(i,ty));
         }
      }

      /* update current column, and column permutation vector */
      __syncthreads();
      if (tx == 0) {
         As(ty,j) = (ty == j) ? diagRcp : negOp (mulOp (As(ty,j), diagRcp));
         if (ty == j) {
            icol[j] = perm[j];
         }
      }

      j++;
   } while (j < N);

   __syncthreads();
   for (int i = tx; i < N; i += blockDim.x) {
      Ainv[icol[ty] * N + i] = As(i,ty);
   }
}

template <typename T, int arch>
int matinv_gje3 (const T *A_d, T *Ainv_d, int n, int batch)
{
   typedef void (* func)(const T *A_d, T *Ainv_d, int n, int batch);

   static int padding[78] = {
      config<T,arch>::gje3Pad_00, config<T,arch>::gje3Pad_01,
      config<T,arch>::gje3Pad_02, config<T,arch>::gje3Pad_03,
      config<T,arch>::gje3Pad_04, config<T,arch>::gje3Pad_05,
      config<T,arch>::gje3Pad_06, config<T,arch>::gje3Pad_07,
      config<T,arch>::gje3Pad_08, config<T,arch>::gje3Pad_09,
      config<T,arch>::gje3Pad_10, config<T,arch>::gje3Pad_11,
      config<T,arch>::gje3Pad_12, config<T,arch>::gje3Pad_13,
      config<T,arch>::gje3Pad_14, config<T,arch>::gje3Pad_15,
      config<T,arch>::gje3Pad_16, config<T,arch>::gje3Pad_17,
      config<T,arch>::gje3Pad_18, config<T,arch>::gje3Pad_19,
      config<T,arch>::gje3Pad_20, config<T,arch>::gje3Pad_21,
      config<T,arch>::gje3Pad_22, config<T,arch>::gje3Pad_23,
      config<T,arch>::gje3Pad_24, config<T,arch>::gje3Pad_25,
      config<T,arch>::gje3Pad_26, config<T,arch>::gje3Pad_27,
      config<T,arch>::gje3Pad_28, config<T,arch>::gje3Pad_29,
      config<T,arch>::gje3Pad_30, config<T,arch>::gje3Pad_31,
      config<T,arch>::gje3Pad_32, config<T,arch>::gje3Pad_33,
      config<T,arch>::gje3Pad_34, config<T,arch>::gje3Pad_35,
      config<T,arch>::gje3Pad_36, config<T,arch>::gje3Pad_37,
      config<T,arch>::gje3Pad_38, config<T,arch>::gje3Pad_39,
      config<T,arch>::gje3Pad_40, config<T,arch>::gje3Pad_41,
      config<T,arch>::gje3Pad_42, config<T,arch>::gje3Pad_43,
      config<T,arch>::gje3Pad_44, config<T,arch>::gje3Pad_45,
      config<T,arch>::gje3Pad_46, config<T,arch>::gje3Pad_47,
      config<T,arch>::gje3Pad_48, config<T,arch>::gje3Pad_49,
      config<T,arch>::gje3Pad_50, config<T,arch>::gje3Pad_51,
      config<T,arch>::gje3Pad_52, config<T,arch>::gje3Pad_53,
      config<T,arch>::gje3Pad_54, config<T,arch>::gje3Pad_55,
      config<T,arch>::gje3Pad_56, config<T,arch>::gje3Pad_57,
      config<T,arch>::gje3Pad_58, config<T,arch>::gje3Pad_59,
      config<T,arch>::gje3Pad_60, config<T,arch>::gje3Pad_61,
      config<T,arch>::gje3Pad_62, config<T,arch>::gje3Pad_63,
      config<T,arch>::gje3Pad_64, config<T,arch>::gje3Pad_65,
      config<T,arch>::gje3Pad_66, config<T,arch>::gje3Pad_67,
      config<T,arch>::gje3Pad_68, config<T,arch>::gje3Pad_69,
      config<T,arch>::gje3Pad_70, config<T,arch>::gje3Pad_71,
      config<T,arch>::gje3Pad_72, config<T,arch>::gje3Pad_73,
      config<T,arch>::gje3Pad_74, config<T,arch>::gje3Pad_75,
      config<T,arch>::gje3Pad_76, config<T,arch>::gje3Pad_77
   };
   static int dimX[78] = {
      config<T,arch>::gje3DimX_00, config<T,arch>::gje3DimX_01, 
      config<T,arch>::gje3DimX_02, config<T,arch>::gje3DimX_03, 
      config<T,arch>::gje3DimX_04, config<T,arch>::gje3DimX_05, 
      config<T,arch>::gje3DimX_06, config<T,arch>::gje3DimX_07, 
      config<T,arch>::gje3DimX_08, config<T,arch>::gje3DimX_09, 
      config<T,arch>::gje3DimX_10, config<T,arch>::gje3DimX_11, 
      config<T,arch>::gje3DimX_12, config<T,arch>::gje3DimX_13, 
      config<T,arch>::gje3DimX_14, config<T,arch>::gje3DimX_15, 
      config<T,arch>::gje3DimX_16, config<T,arch>::gje3DimX_17, 
      config<T,arch>::gje3DimX_18, config<T,arch>::gje3DimX_19, 
      config<T,arch>::gje3DimX_20, config<T,arch>::gje3DimX_21, 
      config<T,arch>::gje3DimX_22, config<T,arch>::gje3DimX_23, 
      config<T,arch>::gje3DimX_24, config<T,arch>::gje3DimX_25, 
      config<T,arch>::gje3DimX_26, config<T,arch>::gje3DimX_27, 
      config<T,arch>::gje3DimX_28, config<T,arch>::gje3DimX_29, 
      config<T,arch>::gje3DimX_30, config<T,arch>::gje3DimX_31, 
      config<T,arch>::gje3DimX_32, config<T,arch>::gje3DimX_33, 
      config<T,arch>::gje3DimX_34, config<T,arch>::gje3DimX_35, 
      config<T,arch>::gje3DimX_36, config<T,arch>::gje3DimX_37, 
      config<T,arch>::gje3DimX_38, config<T,arch>::gje3DimX_39, 
      config<T,arch>::gje3DimX_40, config<T,arch>::gje3DimX_41, 
      config<T,arch>::gje3DimX_42, config<T,arch>::gje3DimX_43, 
      config<T,arch>::gje3DimX_44, config<T,arch>::gje3DimX_45, 
      config<T,arch>::gje3DimX_46, config<T,arch>::gje3DimX_47, 
      config<T,arch>::gje3DimX_48, config<T,arch>::gje3DimX_49, 
      config<T,arch>::gje3DimX_50, config<T,arch>::gje3DimX_51, 
      config<T,arch>::gje3DimX_52, config<T,arch>::gje3DimX_53, 
      config<T,arch>::gje3DimX_54, config<T,arch>::gje3DimX_55,
      config<T,arch>::gje3DimX_56, config<T,arch>::gje3DimX_57,
      config<T,arch>::gje3DimX_58, config<T,arch>::gje3DimX_59,
      config<T,arch>::gje3DimX_60, config<T,arch>::gje3DimX_61,
      config<T,arch>::gje3DimX_62, config<T,arch>::gje3DimX_63,
      config<T,arch>::gje3DimX_64, config<T,arch>::gje3DimX_65,
      config<T,arch>::gje3DimX_66, config<T,arch>::gje3DimX_67,
      config<T,arch>::gje3DimX_68, config<T,arch>::gje3DimX_69,
      config<T,arch>::gje3DimX_70, config<T,arch>::gje3DimX_71,
      config<T,arch>::gje3DimX_72, config<T,arch>::gje3DimX_73,
      config<T,arch>::gje3DimX_74, config<T,arch>::gje3DimX_75,
      config<T,arch>::gje3DimX_76, config<T,arch>::gje3DimX_77
   };
   static int srchThrd[78] = { 
      config<T,arch>::gje3SrchThrd_00, config<T,arch>::gje3SrchThrd_01,
      config<T,arch>::gje3SrchThrd_02, config<T,arch>::gje3SrchThrd_03,
      config<T,arch>::gje3SrchThrd_04, config<T,arch>::gje3SrchThrd_05,
      config<T,arch>::gje3SrchThrd_06, config<T,arch>::gje3SrchThrd_07,
      config<T,arch>::gje3SrchThrd_08, config<T,arch>::gje3SrchThrd_09,
      config<T,arch>::gje3SrchThrd_10, config<T,arch>::gje3SrchThrd_11,  
      config<T,arch>::gje3SrchThrd_12, config<T,arch>::gje3SrchThrd_13,
      config<T,arch>::gje3SrchThrd_14, config<T,arch>::gje3SrchThrd_15,
      config<T,arch>::gje3SrchThrd_16, config<T,arch>::gje3SrchThrd_17,
      config<T,arch>::gje3SrchThrd_18, config<T,arch>::gje3SrchThrd_19,
      config<T,arch>::gje3SrchThrd_20, config<T,arch>::gje3SrchThrd_21,
      config<T,arch>::gje3SrchThrd_22, config<T,arch>::gje3SrchThrd_23,
      config<T,arch>::gje3SrchThrd_24, config<T,arch>::gje3SrchThrd_25,
      config<T,arch>::gje3SrchThrd_26, config<T,arch>::gje3SrchThrd_27,
      config<T,arch>::gje3SrchThrd_28, config<T,arch>::gje3SrchThrd_29,
      config<T,arch>::gje3SrchThrd_30, config<T,arch>::gje3SrchThrd_31,
      config<T,arch>::gje3SrchThrd_32, config<T,arch>::gje3SrchThrd_33,
      config<T,arch>::gje3SrchThrd_34, config<T,arch>::gje3SrchThrd_35,
      config<T,arch>::gje3SrchThrd_36, config<T,arch>::gje3SrchThrd_37,
      config<T,arch>::gje3SrchThrd_38, config<T,arch>::gje3SrchThrd_39,
      config<T,arch>::gje3SrchThrd_40, config<T,arch>::gje3SrchThrd_41,  
      config<T,arch>::gje3SrchThrd_42, config<T,arch>::gje3SrchThrd_43,
      config<T,arch>::gje3SrchThrd_44, config<T,arch>::gje3SrchThrd_45,
      config<T,arch>::gje3SrchThrd_46, config<T,arch>::gje3SrchThrd_47,
      config<T,arch>::gje3SrchThrd_48, config<T,arch>::gje3SrchThrd_49,
      config<T,arch>::gje3SrchThrd_50, config<T,arch>::gje3SrchThrd_51,
      config<T,arch>::gje3SrchThrd_52, config<T,arch>::gje3SrchThrd_53,
      config<T,arch>::gje3SrchThrd_54, config<T,arch>::gje3SrchThrd_55,
      config<T,arch>::gje3SrchThrd_56, config<T,arch>::gje3SrchThrd_57,
      config<T,arch>::gje3SrchThrd_58, config<T,arch>::gje3SrchThrd_59,
      config<T,arch>::gje3SrchThrd_60, config<T,arch>::gje3SrchThrd_61,
      config<T,arch>::gje3SrchThrd_62, config<T,arch>::gje3SrchThrd_63,
      config<T,arch>::gje3SrchThrd_64, config<T,arch>::gje3SrchThrd_65,
      config<T,arch>::gje3SrchThrd_66, config<T,arch>::gje3SrchThrd_67,
      config<T,arch>::gje3SrchThrd_68, config<T,arch>::gje3SrchThrd_69,
      config<T,arch>::gje3SrchThrd_70, config<T,arch>::gje3SrchThrd_71,
      config<T,arch>::gje3SrchThrd_72, config<T,arch>::gje3SrchThrd_73,
      config<T,arch>::gje3SrchThrd_74, config<T,arch>::gje3SrchThrd_75,
      config<T,arch>::gje3SrchThrd_76, config<T,arch>::gje3SrchThrd_77
   };

   func pf[78] = {
      0,       
      0,              
      matinv_gje3<T, config<T,arch>::gje3Pad_02, config<T,arch>::gje3SrchThrd_02, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_03, config<T,arch>::gje3SrchThrd_03, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_04, config<T,arch>::gje3SrchThrd_04, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_05, config<T,arch>::gje3SrchThrd_05, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_06, config<T,arch>::gje3SrchThrd_06, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_07, config<T,arch>::gje3SrchThrd_07, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_08, config<T,arch>::gje3SrchThrd_08, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_09, config<T,arch>::gje3SrchThrd_09, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_10, config<T,arch>::gje3SrchThrd_10, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_11, config<T,arch>::gje3SrchThrd_11, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_12, config<T,arch>::gje3SrchThrd_12, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_13, config<T,arch>::gje3SrchThrd_13, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_14, config<T,arch>::gje3SrchThrd_14, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_15, config<T,arch>::gje3SrchThrd_15, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_16, config<T,arch>::gje3SrchThrd_16, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_17, config<T,arch>::gje3SrchThrd_17, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_18, config<T,arch>::gje3SrchThrd_18, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_19, config<T,arch>::gje3SrchThrd_19, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_20, config<T,arch>::gje3SrchThrd_20, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_21, config<T,arch>::gje3SrchThrd_21, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_22, config<T,arch>::gje3SrchThrd_22, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_23, config<T,arch>::gje3SrchThrd_23, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_24, config<T,arch>::gje3SrchThrd_24, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_25, config<T,arch>::gje3SrchThrd_25, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_26, config<T,arch>::gje3SrchThrd_26, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_27, config<T,arch>::gje3SrchThrd_27, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_28, config<T,arch>::gje3SrchThrd_28, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_29, config<T,arch>::gje3SrchThrd_29, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_30, config<T,arch>::gje3SrchThrd_30, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_31, config<T,arch>::gje3SrchThrd_31, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_32, config<T,arch>::gje3SrchThrd_32, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_33, config<T,arch>::gje3SrchThrd_33, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_34, config<T,arch>::gje3SrchThrd_34, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_35, config<T,arch>::gje3SrchThrd_35, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_36, config<T,arch>::gje3SrchThrd_36, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_37, config<T,arch>::gje3SrchThrd_37, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_38, config<T,arch>::gje3SrchThrd_38, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_39, config<T,arch>::gje3SrchThrd_39, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_40, config<T,arch>::gje3SrchThrd_40, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_41, config<T,arch>::gje3SrchThrd_41, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_42, config<T,arch>::gje3SrchThrd_42, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_43, config<T,arch>::gje3SrchThrd_43, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_44, config<T,arch>::gje3SrchThrd_44, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_45, config<T,arch>::gje3SrchThrd_45, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_46, config<T,arch>::gje3SrchThrd_46, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_47, config<T,arch>::gje3SrchThrd_47, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_48, config<T,arch>::gje3SrchThrd_48, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_49, config<T,arch>::gje3SrchThrd_49, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_50, config<T,arch>::gje3SrchThrd_50, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_51, config<T,arch>::gje3SrchThrd_51, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_52, config<T,arch>::gje3SrchThrd_52, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_53, config<T,arch>::gje3SrchThrd_53, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_54, config<T,arch>::gje3SrchThrd_54, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_55, config<T,arch>::gje3SrchThrd_55, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_56, config<T,arch>::gje3SrchThrd_56, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_57, config<T,arch>::gje3SrchThrd_57, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_58, config<T,arch>::gje3SrchThrd_58, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_59, config<T,arch>::gje3SrchThrd_59, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_60, config<T,arch>::gje3SrchThrd_60, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_61, config<T,arch>::gje3SrchThrd_61, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_62, config<T,arch>::gje3SrchThrd_62, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_63, config<T,arch>::gje3SrchThrd_63, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_64, config<T,arch>::gje3SrchThrd_64, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_65, config<T,arch>::gje3SrchThrd_65, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_66, config<T,arch>::gje3SrchThrd_66, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_67, config<T,arch>::gje3SrchThrd_67, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_68, config<T,arch>::gje3SrchThrd_68, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_69, config<T,arch>::gje3SrchThrd_69, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_70, config<T,arch>::gje3SrchThrd_70, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_71, config<T,arch>::gje3SrchThrd_71, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_72, config<T,arch>::gje3SrchThrd_72, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_73, config<T,arch>::gje3SrchThrd_73, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_74, config<T,arch>::gje3SrchThrd_74, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_75, config<T,arch>::gje3SrchThrd_75, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_76, config<T,arch>::gje3SrchThrd_76, arch>,
      matinv_gje3<T, config<T,arch>::gje3Pad_77, config<T,arch>::gje3SrchThrd_77, arch>
   };

   if (n < config<T,arch>::gje3MinDim || n > config<T,arch>::gje3MaxDim ||
      batch < 1) {
         return -1;
   }

   dim3 dimBlock(dimX[n], n);
   dim3 dimGrid;
   if (batch <= GRID_DIM_LIMIT) {
      dimGrid.x = batch;
      dimGrid.y = 1;
      dimGrid.z = 1;
   } else {
      dimGrid.x = GRID_DIM_LIMIT;
      dimGrid.y = (batch + GRID_DIM_LIMIT-1) / GRID_DIM_LIMIT;
      dimGrid.z = 1;
   }
   int smem_size = (sizeof(A_d[0]) * (n + padding[n]) * (n) +
      sizeof(T) * srchThrd[n] +
      sizeof(int) * srchThrd[n] +
      sizeof(int) * 2 * n);
   pf[n]<<<dimGrid,dimBlock,smem_size>>>(A_d,Ainv_d,n,batch);
   cudaError_t err = cudaGetLastError();
   /* Check synchronous errors, i.e. pre-launch */
   if (cudaSuccess != err) {
      return -2;
   }
   return 0;
}

/* C callable wrapper functions */

int dmatinv_batch (double *A, double *Ainv, int n, int batch)
{ 
   return matinv_gje3<double,GPU_ARCH>(A, Ainv, n, batch);
}

int zmatinv_batch (cuDoubleComplex *A, cuDoubleComplex *Ainv, int n, int batch)
{ 
   return matinv_gje3<cuDoubleComplex,GPU_ARCH>(A, Ainv, n, batch);
}

int fmatinv_batch (float *A, float *Ainv, int n, int batch)
{ 
   return matinv_gje3<float,GPU_ARCH>(A, Ainv, n, batch);
}

int zfmatinv_batch (cuComplex *A, cuComplex *Ainv, int n, int batch)
{ 
   return matinv_gje3<cuComplex,GPU_ARCH>(A, Ainv, n, batch);
}


