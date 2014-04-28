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
#ifndef MEMORY_ONLY
#include "cuComplex.h"
#else
#include "cuComplex_memonly.h"
#endif
#include "nvidia_solve.h"
#ifndef MEMORY_ONLY
#include "operations.h"
#else
#include "operations_memonly.h"
#endif



#define GRID_DIM_LIMIT  (65520)
#define PIVOT_THRDS     (2)

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

// float is currently just a copy of the double config
template<> class config<float,ARCH_SM13> {
public:
   enum { fsMinDim   =    2 };
   enum { fsMaxDim   =   44 };

   enum { ge2MinBlks =    1 };
   enum { ge2MaxThrds=  768 }; /* sm_13, 20 registers per thread */
   enum { gj1MinBlks =    1 };
   enum { gj1MaxThrds=  768 }; /* sm_13, 20 registers per thread */
   enum { gj2MinBlks =    1 };
   enum { gj2MaxThrds=  768 }; /* sm_13, 20 registers per thread */

   enum { fsDimX_00  =   -1 };
   enum { fsDimX_01  =   -1 };
   enum { fsDimX_02  =    2 };
   enum { fsDimX_03  =    3 };
   enum { fsDimX_04  =    2 };
   enum { fsDimX_05  =    2 };
   enum { fsDimX_06  =    2 };
   enum { fsDimX_07  =    2 };
   enum { fsDimX_08  =    3 };
   enum { fsDimX_09  =    3 };
   enum { fsDimX_10  =    2 };
   enum { fsDimX_11  =    2 };
   enum { fsDimX_12  =    2 };
   enum { fsDimX_13  =    2 };
   enum { fsDimX_14  =    2 };
   enum { fsDimX_15  =    2 };
   enum { fsDimX_16  =    2 };
   enum { fsDimX_17  =    2 };
   enum { fsDimX_18  =    2 };
   enum { fsDimX_19  =    3 };
   enum { fsDimX_20  =    3 };
   enum { fsDimX_21  =    2 };
   enum { fsDimX_22  =    4 };
   enum { fsDimX_23  =    2 };
   enum { fsDimX_24  =    3 };
   enum { fsDimX_25  =    3 };
   enum { fsDimX_26  =    4 };
   enum { fsDimX_27  =    4 };
   enum { fsDimX_28  =    4 };
   enum { fsDimX_29  =    5 };
   enum { fsDimX_30  =    4 };
   enum { fsDimX_31  =    5 };
   enum { fsDimX_32  =    8 };
   enum { fsDimX_33  =    8 };
   enum { fsDimX_34  =    5 };
   enum { fsDimX_35  =    8 };
   enum { fsDimX_36  =    8 };
   enum { fsDimX_37  =    5 };
   enum { fsDimX_38  =    6 };
   enum { fsDimX_39  =    6 };
   enum { fsDimX_40  =    8 };
   enum { fsDimX_41  =    8 };
   enum { fsDimX_42  =    4 };
   enum { fsDimX_43  =    6 };
   enum { fsDimX_44  =    8 };
   enum { fsDimX_45  =   -1 };
   enum { fsDimX_46  =   -1 };
   enum { fsDimX_47  =   -1 };
   enum { fsDimX_48  =   -1 };
   enum { fsDimX_49  =   -1 };
   enum { fsDimX_50  =   -1 };
   enum { fsDimX_51  =   -1 };
   enum { fsDimX_52  =   -1 };
   enum { fsDimX_53  =   -1 };
   enum { fsDimX_54  =   -1 };
   enum { fsDimX_55  =   -1 };
   enum { fsDimX_56  =   -1 };
   enum { fsDimX_57  =   -1 };
   enum { fsDimX_58  =   -1 };
   enum { fsDimX_59  =   -1 };
   enum { fsDimX_60  =   -1 };
   enum { fsDimX_61  =   -1 };
   enum { fsDimX_62  =   -1 };
   enum { fsDimX_63  =   -1 };
   enum { fsDimX_64  =   -1 };
   enum { fsDimX_65  =   -1 };
   enum { fsDimX_66  =   -1 };
   enum { fsDimX_67  =   -1 };
   enum { fsDimX_68  =   -1 };
   enum { fsDimX_69  =   -1 };
   enum { fsDimX_70  =   -1 };
   enum { fsDimX_71  =   -1 };
   enum { fsDimX_72  =   -1 };
   enum { fsDimX_73  =   -1 };
   enum { fsDimX_74  =   -1 };
   enum { fsDimX_75  =   -1 };
   enum { fsDimX_76  =   -1 };

   enum { fsPad_00   =   -1 };
   enum { fsPad_01   =   -1 };
   enum { fsPad_02   =    0 };
   enum { fsPad_03   =    0 };    
   enum { fsPad_04   =    0 };
   enum { fsPad_05   =    0 };
   enum { fsPad_06   =    1 };
   enum { fsPad_07   =    4 };
   enum { fsPad_08   =    3 };
   enum { fsPad_09   =    2 };
   enum { fsPad_10   =    1 };
   enum { fsPad_11   =    2 };
   enum { fsPad_12   =    1 };
   enum { fsPad_13   =    2 };
   enum { fsPad_14   =    1 };
   enum { fsPad_15   =    0 };
   enum { fsPad_16   =    1 };
   enum { fsPad_17   =    0 };
   enum { fsPad_18   =    1 };
   enum { fsPad_19   =    2 };
   enum { fsPad_20   =    1 };
   enum { fsPad_21   =    0 };
   enum { fsPad_22   =    4 };
   enum { fsPad_23   =    2 };
   enum { fsPad_24   =    1 };
   enum { fsPad_25   =    2 };
   enum { fsPad_26   =    4 };
   enum { fsPad_27   =    3 };
   enum { fsPad_28   =    2 };
   enum { fsPad_29   =    0 };
   enum { fsPad_30   =    0 };
   enum { fsPad_31   =    0 };
   enum { fsPad_32   =    4 };
   enum { fsPad_33   =    3 };
   enum { fsPad_34   =    3 };
   enum { fsPad_35   =    1 };
   enum { fsPad_36   =    1 };
   enum { fsPad_37   =    0 };
   enum { fsPad_38   =    0 };
   enum { fsPad_39   =    4 };
   enum { fsPad_40   =    3 };
   enum { fsPad_41   =    3 };
   enum { fsPad_42   =    4 };
   enum { fsPad_43   =    3 };
   enum { fsPad_44   =    1 };
   enum { fsPad_45   =   -1 };
   enum { fsPad_46   =   -1 };
   enum { fsPad_47   =   -1 };
   enum { fsPad_48   =   -1 };
   enum { fsPad_49   =   -1 };
   enum { fsPad_50   =   -1 };
   enum { fsPad_51   =   -1 };
   enum { fsPad_52   =   -1 };
   enum { fsPad_53   =   -1 };
   enum { fsPad_54   =   -1 };
   enum { fsPad_55   =   -1 };
   enum { fsPad_56   =   -1 };
   enum { fsPad_57   =   -1 };
   enum { fsPad_58   =   -1 };
   enum { fsPad_59   =   -1 };
   enum { fsPad_60   =   -1 };
   enum { fsPad_61   =   -1 };
   enum { fsPad_62   =   -1 };
   enum { fsPad_63   =   -1 };
   enum { fsPad_64   =   -1 };
   enum { fsPad_65   =   -1 };
   enum { fsPad_66   =   -1 };
   enum { fsPad_67   =   -1 };
   enum { fsPad_68   =   -1 };
   enum { fsPad_69   =   -1 };
   enum { fsPad_70   =   -1 };
   enum { fsPad_71   =   -1 };
   enum { fsPad_72   =   -1 };
   enum { fsPad_73   =   -1 };
   enum { fsPad_74   =   -1 };
   enum { fsPad_75   =   -1 };
   enum { fsPad_76   =   -1 };

   enum { fsSrchThrd_00   =   -1 };
   enum { fsSrchThrd_01   =   -1 };
   enum { fsSrchThrd_02   =    2 };
   enum { fsSrchThrd_03   =    2 };    
   enum { fsSrchThrd_04   =    2 };
   enum { fsSrchThrd_05   =    2 };
   enum { fsSrchThrd_06   =    2 };
   enum { fsSrchThrd_07   =    2 };
   enum { fsSrchThrd_08   =    2 };
   enum { fsSrchThrd_09   =    2 };
   enum { fsSrchThrd_10   =    2 };
   enum { fsSrchThrd_11   =    2 };
   enum { fsSrchThrd_12   =    2 };
   enum { fsSrchThrd_13   =    2 };
   enum { fsSrchThrd_14   =    2 };
   enum { fsSrchThrd_15   =    2 };
   enum { fsSrchThrd_16   =    2 };
   enum { fsSrchThrd_17   =    2 };
   enum { fsSrchThrd_18   =    2 };
   enum { fsSrchThrd_19   =    2 };
   enum { fsSrchThrd_20   =    3 };
   enum { fsSrchThrd_21   =    3 };
   enum { fsSrchThrd_22   =    3 };
   enum { fsSrchThrd_23   =    3 };
   enum { fsSrchThrd_24   =    3 };
   enum { fsSrchThrd_25   =    3 };
   enum { fsSrchThrd_26   =    3 };
   enum { fsSrchThrd_27   =    3 };
   enum { fsSrchThrd_28   =    3 };
   enum { fsSrchThrd_29   =    3 };
   enum { fsSrchThrd_30   =    3 };
   enum { fsSrchThrd_31   =    3 };
   enum { fsSrchThrd_32   =    4 };
   enum { fsSrchThrd_33   =    4 };
   enum { fsSrchThrd_34   =    4 };
   enum { fsSrchThrd_35   =    4 };
   enum { fsSrchThrd_36   =    4 };
   enum { fsSrchThrd_37   =    4 };
   enum { fsSrchThrd_38   =    4 };
   enum { fsSrchThrd_39   =    4 };
   enum { fsSrchThrd_40   =    4 };
   enum { fsSrchThrd_41   =    4 };
   enum { fsSrchThrd_42   =    4 };
   enum { fsSrchThrd_43   =    4 };
   enum { fsSrchThrd_44   =    4 };
   enum { fsSrchThrd_45   =   -1 };
   enum { fsSrchThrd_46   =   -1 };
   enum { fsSrchThrd_47   =   -1 };
   enum { fsSrchThrd_48   =   -1 };
   enum { fsSrchThrd_49   =   -1 };
   enum { fsSrchThrd_50   =   -1 };
   enum { fsSrchThrd_51   =   -1 };
   enum { fsSrchThrd_52   =   -1 };
   enum { fsSrchThrd_53   =   -1 };
   enum { fsSrchThrd_54   =   -1 };
   enum { fsSrchThrd_55   =   -1 };
   enum { fsSrchThrd_56   =   -1 };
   enum { fsSrchThrd_57   =   -1 };
   enum { fsSrchThrd_58   =   -1 };
   enum { fsSrchThrd_59   =   -1 };
   enum { fsSrchThrd_60   =   -1 };
   enum { fsSrchThrd_61   =   -1 };
   enum { fsSrchThrd_62   =   -1 };
   enum { fsSrchThrd_63   =   -1 };
   enum { fsSrchThrd_64   =   -1 };
   enum { fsSrchThrd_65   =   -1 };
   enum { fsSrchThrd_66   =   -1 };
   enum { fsSrchThrd_67   =   -1 };
   enum { fsSrchThrd_68   =   -1 };
   enum { fsSrchThrd_69   =   -1 };
   enum { fsSrchThrd_70   =   -1 };
   enum { fsSrchThrd_71   =   -1 };
   enum { fsSrchThrd_72   =   -1 };
   enum { fsSrchThrd_73   =   -1 };
   enum { fsSrchThrd_74   =   -1 };
   enum { fsSrchThrd_75   =   -1 };
   enum { fsSrchThrd_76   =   -1 };
};

template<> class config<float,ARCH_SM20> {
public:
   enum { fsMinDim   =    2 };
   enum { fsMaxDim   =   76 };

   enum { ge2MinBlks =    1 };
   enum { ge2MaxThrds= 1280 }; /* sm_2x, 24 registers per thread */
   enum { gj1MinBlks =    1 };
   enum { gj1MaxThrds= 1536 }; /* sm_2x, 20 registers per thread */
   enum { gj2MinBlks =    1 };
   enum { gj2MaxThrds= 1536 }; /* sm_2x, 20 registers per thread */

   enum { fsDimX_00  =   -1 };
   enum { fsDimX_01  =   -1 };
   enum { fsDimX_02  =    2 };
   enum { fsDimX_03  =    3 };
   enum { fsDimX_04  =    4 };
   enum { fsDimX_05  =    5 };
   enum { fsDimX_06  =    6 };
   enum { fsDimX_07  =    7 };
   enum { fsDimX_08  =    8 };
   enum { fsDimX_09  =    9 };
   enum { fsDimX_10  =    5 };
   enum { fsDimX_11  =    5 };
   enum { fsDimX_12  =    4 };
   enum { fsDimX_13  =    4 };
   enum { fsDimX_14  =    4 };
   enum { fsDimX_15  =    4 };
   enum { fsDimX_16  =    3 };
   enum { fsDimX_17  =    3 };
   enum { fsDimX_18  =    3 };
   enum { fsDimX_19  =    3 };
   enum { fsDimX_20  =    3 };
   enum { fsDimX_21  =    4 };
   enum { fsDimX_22  =    4 };
   enum { fsDimX_23  =    4 };
   enum { fsDimX_24  =    2 };
   enum { fsDimX_25  =    2 };
   enum { fsDimX_26  =    2 };
   enum { fsDimX_27  =    2 };
   enum { fsDimX_28  =    3 };
   enum { fsDimX_29  =    3 };
   enum { fsDimX_30  =    3 };
   enum { fsDimX_31  =    3 };
   enum { fsDimX_32  =    3 };
   enum { fsDimX_33  =    3 };
   enum { fsDimX_34  =    4 };
   enum { fsDimX_35  =    4 };
   enum { fsDimX_36  =    4 };
   enum { fsDimX_37  =    5 };
   enum { fsDimX_38  =    4 };
   enum { fsDimX_39  =    4 };
   enum { fsDimX_40  =    4 };
   enum { fsDimX_41  =    4 };
   enum { fsDimX_42  =    4 };
   enum { fsDimX_43  =    4 };
   enum { fsDimX_44  =    4 };
   enum { fsDimX_45  =    4 };
   enum { fsDimX_46  =    4 };
   enum { fsDimX_47  =    4 };
   enum { fsDimX_48  =    5 };
   enum { fsDimX_49  =    5 };
   enum { fsDimX_50  =    5 };
   enum { fsDimX_51  =    4 };
   enum { fsDimX_52  =    4 };
   enum { fsDimX_53  =    5 };
   enum { fsDimX_54  =    6 };
   enum { fsDimX_55  =    8 };
   enum { fsDimX_56  =    8 };
   enum { fsDimX_57  =    6 };
   enum { fsDimX_58  =    7 };
   enum { fsDimX_59  =    6 };
   enum { fsDimX_60  =    6 };
   enum { fsDimX_61  =    6 };
   enum { fsDimX_62  =    7 };
   enum { fsDimX_63  =    6 };
   enum { fsDimX_64  =    5 };
   enum { fsDimX_65  =    6 };
   enum { fsDimX_66  =    6 };
   enum { fsDimX_67  =    6 };
   enum { fsDimX_68  =    6 };
   enum { fsDimX_69  =    6 };
   enum { fsDimX_70  =    7 };
   enum { fsDimX_71  =    7 };
   enum { fsDimX_72  =    6 };
   enum { fsDimX_73  =    6 };
   enum { fsDimX_74  =    6 };
   enum { fsDimX_75  =    6 };
   enum { fsDimX_76  =    4 };

   enum { fsPad_00   =   -1 };
   enum { fsPad_01   =   -1 };
   enum { fsPad_02   =    0 };
   enum { fsPad_03   =    0 };
   enum { fsPad_04   =    0 };
   enum { fsPad_05   =    0 };
   enum { fsPad_06   =    0 };
   enum { fsPad_07   =    0 };
   enum { fsPad_08   =    0 };
   enum { fsPad_09   =    0 };
   enum { fsPad_10   =    0 };
   enum { fsPad_11   =    1 };
   enum { fsPad_12   =    0 };
   enum { fsPad_13   =    1 };
   enum { fsPad_14   =    0 };
   enum { fsPad_15   =    3 };
   enum { fsPad_16   =    3 };
   enum { fsPad_17   =    2 };
   enum { fsPad_18   =    1 };
   enum { fsPad_19   =    0 };
   enum { fsPad_20   =    2 };
   enum { fsPad_21   =    0 };
   enum { fsPad_22   =    0 };
   enum { fsPad_23   =    0 };
   enum { fsPad_24   =    2 };
   enum { fsPad_25   =    1 };
   enum { fsPad_26   =    0 };
   enum { fsPad_27   =    0 };
   enum { fsPad_28   =    1 };
   enum { fsPad_29   =    1 };
   enum { fsPad_30   =    0 };
   enum { fsPad_31   =    0 };
   enum { fsPad_32   =    3 };
   enum { fsPad_33   =    2 };
   enum { fsPad_34   =    0 };
   enum { fsPad_35   =    1 };
   enum { fsPad_36   =    0 };
   enum { fsPad_37   =    0 };
   enum { fsPad_38   =    1 };
   enum { fsPad_39   =    3 };
   enum { fsPad_40   =    4 };
   enum { fsPad_41   =    3 };
   enum { fsPad_42   =    2 };
   enum { fsPad_43   =    1 };
   enum { fsPad_44   =    0 };
   enum { fsPad_45   =    5 };
   enum { fsPad_46   =    4 };
   enum { fsPad_47   =    3 };
   enum { fsPad_48   =    5 };
   enum { fsPad_49   =    4 };
   enum { fsPad_50   =    3 };
   enum { fsPad_51   =    1 };
   enum { fsPad_52   =    0 };
   enum { fsPad_53   =    0 };
   enum { fsPad_54   =    0 };
   enum { fsPad_55   =    1 };
   enum { fsPad_56   =    0 };
   enum { fsPad_57   =    2 };
   enum { fsPad_58   =    2 };
   enum { fsPad_59   =    1 };
   enum { fsPad_60   =    1 };
   enum { fsPad_61   =    1 };
   enum { fsPad_62   =    1 };
   enum { fsPad_63   =    5 };
   enum { fsPad_64   =    5 };
   enum { fsPad_65   =    5 };
   enum { fsPad_66   =    4 };
   enum { fsPad_67   =    3 };
   enum { fsPad_68   =    2 };
   enum { fsPad_69   =    1 };
   enum { fsPad_70   =    1 };
   enum { fsPad_71   =    0 };
   enum { fsPad_72   =    3 };
   enum { fsPad_73   =    2 };
   enum { fsPad_74   =    1 };
   enum { fsPad_75   =    0 };
   enum { fsPad_76   =    0 };

   enum { fsSrchThrd_00   =   -1 };
   enum { fsSrchThrd_01   =   -1 };
   enum { fsSrchThrd_02   =    2 };
   enum { fsSrchThrd_03   =    2 };    
   enum { fsSrchThrd_04   =    2 };
   enum { fsSrchThrd_05   =    2 };
   enum { fsSrchThrd_06   =    2 };
   enum { fsSrchThrd_07   =    2 };
   enum { fsSrchThrd_08   =    2 };
   enum { fsSrchThrd_09   =    2 };
   enum { fsSrchThrd_10   =    2 };
   enum { fsSrchThrd_11   =    3 };
   enum { fsSrchThrd_12   =    3 };
   enum { fsSrchThrd_13   =    3 };
   enum { fsSrchThrd_14   =    3 };
   enum { fsSrchThrd_15   =    3 };
   enum { fsSrchThrd_16   =    3 };
   enum { fsSrchThrd_17   =    3 };
   enum { fsSrchThrd_18   =    3 };
   enum { fsSrchThrd_19   =    3 };
   enum { fsSrchThrd_20   =    3 };
   enum { fsSrchThrd_21   =    3 };
   enum { fsSrchThrd_22   =    3 };
   enum { fsSrchThrd_23   =    3 };
   enum { fsSrchThrd_24   =    3 };
   enum { fsSrchThrd_25   =    4 };
   enum { fsSrchThrd_26   =    4 };
   enum { fsSrchThrd_27   =    4 };
   enum { fsSrchThrd_28   =    4 };
   enum { fsSrchThrd_29   =    4 };
   enum { fsSrchThrd_30   =    4 };
   enum { fsSrchThrd_31   =    4 };
   enum { fsSrchThrd_32   =    4 };
   enum { fsSrchThrd_33   =    4 };
   enum { fsSrchThrd_34   =    4 };
   enum { fsSrchThrd_35   =    4 };
   enum { fsSrchThrd_36   =    4 };
   enum { fsSrchThrd_37   =    4 };
   enum { fsSrchThrd_38   =    4 };
   enum { fsSrchThrd_39   =    4 };
   enum { fsSrchThrd_40   =    5 };
   enum { fsSrchThrd_41   =    5 };
   enum { fsSrchThrd_42   =    5 };
   enum { fsSrchThrd_43   =    5 };
   enum { fsSrchThrd_44   =    5 };
   enum { fsSrchThrd_45   =    5 };
   enum { fsSrchThrd_46   =    5 };
   enum { fsSrchThrd_47   =    5 };
   enum { fsSrchThrd_48   =    5 };
   enum { fsSrchThrd_49   =    5 };
   enum { fsSrchThrd_50   =    5 };
   enum { fsSrchThrd_51   =    5 };
   enum { fsSrchThrd_52   =    5 };
   enum { fsSrchThrd_53   =    5 };
   enum { fsSrchThrd_54   =    5 };
   enum { fsSrchThrd_55   =    5 };
   enum { fsSrchThrd_56   =    5 };
   enum { fsSrchThrd_57   =    5 };
   enum { fsSrchThrd_58   =    5 };
   enum { fsSrchThrd_59   =    5 };
   enum { fsSrchThrd_60   =    5 };
   enum { fsSrchThrd_61   =    5 };
   enum { fsSrchThrd_62   =    5 };
   enum { fsSrchThrd_63   =    5 };
   enum { fsSrchThrd_64   =    5 };
   enum { fsSrchThrd_65   =    5 };
   enum { fsSrchThrd_66   =    5 };
   enum { fsSrchThrd_67   =    5 };
   enum { fsSrchThrd_68   =    5 };
   enum { fsSrchThrd_69   =    5 };
   enum { fsSrchThrd_70   =    5 };
   enum { fsSrchThrd_71   =    5 };
   enum { fsSrchThrd_72   =    5 };
   enum { fsSrchThrd_73   =    5 };
   enum { fsSrchThrd_74   =    5 };
   enum { fsSrchThrd_75   =    5 };
   enum { fsSrchThrd_76   =    5 };
};

template<> class config<double,ARCH_SM13> {
public:
   enum { fsMinDim   =    2 };
   enum { fsMaxDim   =   44 };

   enum { ge2MinBlks =    1 };
   enum { ge2MaxThrds=  768 }; /* sm_13, 20 registers per thread */
   enum { gj1MinBlks =    1 };
   enum { gj1MaxThrds=  768 }; /* sm_13, 20 registers per thread */
   enum { gj2MinBlks =    1 };
   enum { gj2MaxThrds=  768 }; /* sm_13, 20 registers per thread */

   enum { fsDimX_00  =   -1 };
   enum { fsDimX_01  =   -1 };
   enum { fsDimX_02  =    2 };
   enum { fsDimX_03  =    3 };
   enum { fsDimX_04  =    2 };
   enum { fsDimX_05  =    2 };
   enum { fsDimX_06  =    2 };
   enum { fsDimX_07  =    2 };
   enum { fsDimX_08  =    3 };
   enum { fsDimX_09  =    3 };
   enum { fsDimX_10  =    2 };
   enum { fsDimX_11  =    2 };
   enum { fsDimX_12  =    2 };
   enum { fsDimX_13  =    2 };
   enum { fsDimX_14  =    2 };
   enum { fsDimX_15  =    2 };
   enum { fsDimX_16  =    2 };
   enum { fsDimX_17  =    2 };
   enum { fsDimX_18  =    2 };
   enum { fsDimX_19  =    3 };
   enum { fsDimX_20  =    3 };
   enum { fsDimX_21  =    2 };
   enum { fsDimX_22  =    4 };
   enum { fsDimX_23  =    2 };
   enum { fsDimX_24  =    3 };
   enum { fsDimX_25  =    3 };
   enum { fsDimX_26  =    4 };
   enum { fsDimX_27  =    4 };
   enum { fsDimX_28  =    4 };
   enum { fsDimX_29  =    5 };
   enum { fsDimX_30  =    4 };
   enum { fsDimX_31  =    5 };
   enum { fsDimX_32  =    8 };
   enum { fsDimX_33  =    8 };
   enum { fsDimX_34  =    5 };
   enum { fsDimX_35  =    8 };
   enum { fsDimX_36  =    8 };
   enum { fsDimX_37  =    5 };
   enum { fsDimX_38  =    6 };
   enum { fsDimX_39  =    6 };
   enum { fsDimX_40  =    8 };
   enum { fsDimX_41  =    8 };
   enum { fsDimX_42  =    4 };
   enum { fsDimX_43  =    6 };
   enum { fsDimX_44  =    8 };
   enum { fsDimX_45  =   -1 };
   enum { fsDimX_46  =   -1 };
   enum { fsDimX_47  =   -1 };
   enum { fsDimX_48  =   -1 };
   enum { fsDimX_49  =   -1 };
   enum { fsDimX_50  =   -1 };
   enum { fsDimX_51  =   -1 };
   enum { fsDimX_52  =   -1 };
   enum { fsDimX_53  =   -1 };
   enum { fsDimX_54  =   -1 };
   enum { fsDimX_55  =   -1 };
   enum { fsDimX_56  =   -1 };
   enum { fsDimX_57  =   -1 };
   enum { fsDimX_58  =   -1 };
   enum { fsDimX_59  =   -1 };
   enum { fsDimX_60  =   -1 };
   enum { fsDimX_61  =   -1 };
   enum { fsDimX_62  =   -1 };
   enum { fsDimX_63  =   -1 };
   enum { fsDimX_64  =   -1 };
   enum { fsDimX_65  =   -1 };
   enum { fsDimX_66  =   -1 };
   enum { fsDimX_67  =   -1 };
   enum { fsDimX_68  =   -1 };
   enum { fsDimX_69  =   -1 };
   enum { fsDimX_70  =   -1 };
   enum { fsDimX_71  =   -1 };
   enum { fsDimX_72  =   -1 };
   enum { fsDimX_73  =   -1 };
   enum { fsDimX_74  =   -1 };
   enum { fsDimX_75  =   -1 };
   enum { fsDimX_76  =   -1 };

   enum { fsPad_00   =   -1 };
   enum { fsPad_01   =   -1 };
   enum { fsPad_02   =    0 };
   enum { fsPad_03   =    0 };    
   enum { fsPad_04   =    0 };
   enum { fsPad_05   =    0 };
   enum { fsPad_06   =    1 };
   enum { fsPad_07   =    4 };
   enum { fsPad_08   =    3 };
   enum { fsPad_09   =    2 };
   enum { fsPad_10   =    1 };
   enum { fsPad_11   =    2 };
   enum { fsPad_12   =    1 };
   enum { fsPad_13   =    2 };
   enum { fsPad_14   =    1 };
   enum { fsPad_15   =    0 };
   enum { fsPad_16   =    1 };
   enum { fsPad_17   =    0 };
   enum { fsPad_18   =    1 };
   enum { fsPad_19   =    2 };
   enum { fsPad_20   =    1 };
   enum { fsPad_21   =    0 };
   enum { fsPad_22   =    4 };
   enum { fsPad_23   =    2 };
   enum { fsPad_24   =    1 };
   enum { fsPad_25   =    2 };
   enum { fsPad_26   =    4 };
   enum { fsPad_27   =    3 };
   enum { fsPad_28   =    2 };
   enum { fsPad_29   =    0 };
   enum { fsPad_30   =    0 };
   enum { fsPad_31   =    0 };
   enum { fsPad_32   =    4 };
   enum { fsPad_33   =    3 };
   enum { fsPad_34   =    3 };
   enum { fsPad_35   =    1 };
   enum { fsPad_36   =    1 };
   enum { fsPad_37   =    0 };
   enum { fsPad_38   =    0 };
   enum { fsPad_39   =    4 };
   enum { fsPad_40   =    3 };
   enum { fsPad_41   =    3 };
   enum { fsPad_42   =    4 };
   enum { fsPad_43   =    3 };
   enum { fsPad_44   =    1 };
   enum { fsPad_45   =   -1 };
   enum { fsPad_46   =   -1 };
   enum { fsPad_47   =   -1 };
   enum { fsPad_48   =   -1 };
   enum { fsPad_49   =   -1 };
   enum { fsPad_50   =   -1 };
   enum { fsPad_51   =   -1 };
   enum { fsPad_52   =   -1 };
   enum { fsPad_53   =   -1 };
   enum { fsPad_54   =   -1 };
   enum { fsPad_55   =   -1 };
   enum { fsPad_56   =   -1 };
   enum { fsPad_57   =   -1 };
   enum { fsPad_58   =   -1 };
   enum { fsPad_59   =   -1 };
   enum { fsPad_60   =   -1 };
   enum { fsPad_61   =   -1 };
   enum { fsPad_62   =   -1 };
   enum { fsPad_63   =   -1 };
   enum { fsPad_64   =   -1 };
   enum { fsPad_65   =   -1 };
   enum { fsPad_66   =   -1 };
   enum { fsPad_67   =   -1 };
   enum { fsPad_68   =   -1 };
   enum { fsPad_69   =   -1 };
   enum { fsPad_70   =   -1 };
   enum { fsPad_71   =   -1 };
   enum { fsPad_72   =   -1 };
   enum { fsPad_73   =   -1 };
   enum { fsPad_74   =   -1 };
   enum { fsPad_75   =   -1 };
   enum { fsPad_76   =   -1 };

   enum { fsSrchThrd_00   =   -1 };
   enum { fsSrchThrd_01   =   -1 };
   enum { fsSrchThrd_02   =    2 };
   enum { fsSrchThrd_03   =    2 };    
   enum { fsSrchThrd_04   =    2 };
   enum { fsSrchThrd_05   =    2 };
   enum { fsSrchThrd_06   =    2 };
   enum { fsSrchThrd_07   =    2 };
   enum { fsSrchThrd_08   =    2 };
   enum { fsSrchThrd_09   =    2 };
   enum { fsSrchThrd_10   =    2 };
   enum { fsSrchThrd_11   =    2 };
   enum { fsSrchThrd_12   =    2 };
   enum { fsSrchThrd_13   =    2 };
   enum { fsSrchThrd_14   =    2 };
   enum { fsSrchThrd_15   =    2 };
   enum { fsSrchThrd_16   =    2 };
   enum { fsSrchThrd_17   =    2 };
   enum { fsSrchThrd_18   =    2 };
   enum { fsSrchThrd_19   =    2 };
   enum { fsSrchThrd_20   =    3 };
   enum { fsSrchThrd_21   =    3 };
   enum { fsSrchThrd_22   =    3 };
   enum { fsSrchThrd_23   =    3 };
   enum { fsSrchThrd_24   =    3 };
   enum { fsSrchThrd_25   =    3 };
   enum { fsSrchThrd_26   =    3 };
   enum { fsSrchThrd_27   =    3 };
   enum { fsSrchThrd_28   =    3 };
   enum { fsSrchThrd_29   =    3 };
   enum { fsSrchThrd_30   =    3 };
   enum { fsSrchThrd_31   =    3 };
   enum { fsSrchThrd_32   =    4 };
   enum { fsSrchThrd_33   =    4 };
   enum { fsSrchThrd_34   =    4 };
   enum { fsSrchThrd_35   =    4 };
   enum { fsSrchThrd_36   =    4 };
   enum { fsSrchThrd_37   =    4 };
   enum { fsSrchThrd_38   =    4 };
   enum { fsSrchThrd_39   =    4 };
   enum { fsSrchThrd_40   =    4 };
   enum { fsSrchThrd_41   =    4 };
   enum { fsSrchThrd_42   =    4 };
   enum { fsSrchThrd_43   =    4 };
   enum { fsSrchThrd_44   =    4 };
   enum { fsSrchThrd_45   =   -1 };
   enum { fsSrchThrd_46   =   -1 };
   enum { fsSrchThrd_47   =   -1 };
   enum { fsSrchThrd_48   =   -1 };
   enum { fsSrchThrd_49   =   -1 };
   enum { fsSrchThrd_50   =   -1 };
   enum { fsSrchThrd_51   =   -1 };
   enum { fsSrchThrd_52   =   -1 };
   enum { fsSrchThrd_53   =   -1 };
   enum { fsSrchThrd_54   =   -1 };
   enum { fsSrchThrd_55   =   -1 };
   enum { fsSrchThrd_56   =   -1 };
   enum { fsSrchThrd_57   =   -1 };
   enum { fsSrchThrd_58   =   -1 };
   enum { fsSrchThrd_59   =   -1 };
   enum { fsSrchThrd_60   =   -1 };
   enum { fsSrchThrd_61   =   -1 };
   enum { fsSrchThrd_62   =   -1 };
   enum { fsSrchThrd_63   =   -1 };
   enum { fsSrchThrd_64   =   -1 };
   enum { fsSrchThrd_65   =   -1 };
   enum { fsSrchThrd_66   =   -1 };
   enum { fsSrchThrd_67   =   -1 };
   enum { fsSrchThrd_68   =   -1 };
   enum { fsSrchThrd_69   =   -1 };
   enum { fsSrchThrd_70   =   -1 };
   enum { fsSrchThrd_71   =   -1 };
   enum { fsSrchThrd_72   =   -1 };
   enum { fsSrchThrd_73   =   -1 };
   enum { fsSrchThrd_74   =   -1 };
   enum { fsSrchThrd_75   =   -1 };
   enum { fsSrchThrd_76   =   -1 };
};

template<> class config<double,ARCH_SM20> {
public:
   enum { fsMinDim   =    2 };
   enum { fsMaxDim   =   76 };

   enum { ge2MinBlks =    1 };
   enum { ge2MaxThrds= 1280 }; /* sm_2x, 24 registers per thread */
   enum { gj1MinBlks =    1 };
   enum { gj1MaxThrds= 1536 }; /* sm_2x, 20 registers per thread */
   enum { gj2MinBlks =    1 };
   enum { gj2MaxThrds= 1536 }; /* sm_2x, 20 registers per thread */

   enum { fsDimX_00  =   -1 };
   enum { fsDimX_01  =   -1 };
   enum { fsDimX_02  =    2 };
   enum { fsDimX_03  =    3 };
   enum { fsDimX_04  =    4 };
   enum { fsDimX_05  =    5 };
   enum { fsDimX_06  =    6 };
   enum { fsDimX_07  =    7 };
   enum { fsDimX_08  =    8 };
   enum { fsDimX_09  =    9 };
   enum { fsDimX_10  =    5 };
   enum { fsDimX_11  =    5 };
   enum { fsDimX_12  =    4 };
   enum { fsDimX_13  =    4 };
   enum { fsDimX_14  =    4 };
   enum { fsDimX_15  =    4 };
   enum { fsDimX_16  =    3 };
   enum { fsDimX_17  =    3 };
   enum { fsDimX_18  =    3 };
   enum { fsDimX_19  =    3 };
   enum { fsDimX_20  =    3 };
   enum { fsDimX_21  =    4 };
   enum { fsDimX_22  =    4 };
   enum { fsDimX_23  =    4 };
   enum { fsDimX_24  =    2 };
   enum { fsDimX_25  =    2 };
   enum { fsDimX_26  =    2 };
   enum { fsDimX_27  =    2 };
   enum { fsDimX_28  =    3 };
   enum { fsDimX_29  =    3 };
   enum { fsDimX_30  =    3 };
   enum { fsDimX_31  =    3 };
   enum { fsDimX_32  =    3 };
   enum { fsDimX_33  =    3 };
   enum { fsDimX_34  =    4 };
   enum { fsDimX_35  =    4 };
   enum { fsDimX_36  =    4 };
   enum { fsDimX_37  =    5 };
   enum { fsDimX_38  =    4 };
   enum { fsDimX_39  =    4 };
   enum { fsDimX_40  =    4 };
   enum { fsDimX_41  =    4 };
   enum { fsDimX_42  =    4 };
   enum { fsDimX_43  =    4 };
   enum { fsDimX_44  =    4 };
   enum { fsDimX_45  =    4 };
   enum { fsDimX_46  =    4 };
   enum { fsDimX_47  =    4 };
   enum { fsDimX_48  =    5 };
   enum { fsDimX_49  =    5 };
   enum { fsDimX_50  =    5 };
   enum { fsDimX_51  =    4 };
   enum { fsDimX_52  =    4 };
   enum { fsDimX_53  =    5 };
   enum { fsDimX_54  =    6 };
   enum { fsDimX_55  =    8 };
   enum { fsDimX_56  =    8 };
   enum { fsDimX_57  =    6 };
   enum { fsDimX_58  =    7 };
   enum { fsDimX_59  =    6 };
   enum { fsDimX_60  =    6 };
   enum { fsDimX_61  =    6 };
   enum { fsDimX_62  =    7 };
   enum { fsDimX_63  =    6 };
   enum { fsDimX_64  =    5 };
   enum { fsDimX_65  =    6 };
   enum { fsDimX_66  =    6 };
   enum { fsDimX_67  =    6 };
   enum { fsDimX_68  =    6 };
   enum { fsDimX_69  =    6 };
   enum { fsDimX_70  =    7 };
   enum { fsDimX_71  =    7 };
   enum { fsDimX_72  =    6 };
   enum { fsDimX_73  =    6 };
   enum { fsDimX_74  =    6 };
   enum { fsDimX_75  =    6 };
   enum { fsDimX_76  =    4 };

   enum { fsPad_00   =   -1 };
   enum { fsPad_01   =   -1 };
   enum { fsPad_02   =    0 };
   enum { fsPad_03   =    0 };
   enum { fsPad_04   =    0 };
   enum { fsPad_05   =    0 };
   enum { fsPad_06   =    0 };
   enum { fsPad_07   =    0 };
   enum { fsPad_08   =    0 };
   enum { fsPad_09   =    0 };
   enum { fsPad_10   =    0 };
   enum { fsPad_11   =    1 };
   enum { fsPad_12   =    0 };
   enum { fsPad_13   =    1 };
   enum { fsPad_14   =    0 };
   enum { fsPad_15   =    3 };
   enum { fsPad_16   =    3 };
   enum { fsPad_17   =    2 };
   enum { fsPad_18   =    1 };
   enum { fsPad_19   =    0 };
   enum { fsPad_20   =    2 };
   enum { fsPad_21   =    0 };
   enum { fsPad_22   =    0 };
   enum { fsPad_23   =    0 };
   enum { fsPad_24   =    2 };
   enum { fsPad_25   =    1 };
   enum { fsPad_26   =    0 };
   enum { fsPad_27   =    0 };
   enum { fsPad_28   =    1 };
   enum { fsPad_29   =    1 };
   enum { fsPad_30   =    0 };
   enum { fsPad_31   =    0 };
   enum { fsPad_32   =    3 };
   enum { fsPad_33   =    2 };
   enum { fsPad_34   =    0 };
   enum { fsPad_35   =    1 };
   enum { fsPad_36   =    0 };
   enum { fsPad_37   =    0 };
   enum { fsPad_38   =    1 };
   enum { fsPad_39   =    3 };
   enum { fsPad_40   =    4 };
   enum { fsPad_41   =    3 };
   enum { fsPad_42   =    2 };
   enum { fsPad_43   =    1 };
   enum { fsPad_44   =    0 };
   enum { fsPad_45   =    5 };
   enum { fsPad_46   =    4 };
   enum { fsPad_47   =    3 };
   enum { fsPad_48   =    5 };
   enum { fsPad_49   =    4 };
   enum { fsPad_50   =    3 };
   enum { fsPad_51   =    1 };
   enum { fsPad_52   =    0 };
   enum { fsPad_53   =    0 };
   enum { fsPad_54   =    0 };
   enum { fsPad_55   =    1 };
   enum { fsPad_56   =    0 };
   enum { fsPad_57   =    2 };
   enum { fsPad_58   =    2 };
   enum { fsPad_59   =    1 };
   enum { fsPad_60   =    1 };
   enum { fsPad_61   =    1 };
   enum { fsPad_62   =    1 };
   enum { fsPad_63   =    5 };
   enum { fsPad_64   =    5 };
   enum { fsPad_65   =    5 };
   enum { fsPad_66   =    4 };
   enum { fsPad_67   =    3 };
   enum { fsPad_68   =    2 };
   enum { fsPad_69   =    1 };
   enum { fsPad_70   =    1 };
   enum { fsPad_71   =    0 };
   enum { fsPad_72   =    3 };
   enum { fsPad_73   =    2 };
   enum { fsPad_74   =    1 };
   enum { fsPad_75   =    0 };
   enum { fsPad_76   =    0 };

   enum { fsSrchThrd_00   =   -1 };
   enum { fsSrchThrd_01   =   -1 };
   enum { fsSrchThrd_02   =    2 };
   enum { fsSrchThrd_03   =    2 };    
   enum { fsSrchThrd_04   =    2 };
   enum { fsSrchThrd_05   =    2 };
   enum { fsSrchThrd_06   =    2 };
   enum { fsSrchThrd_07   =    2 };
   enum { fsSrchThrd_08   =    2 };
   enum { fsSrchThrd_09   =    2 };
   enum { fsSrchThrd_10   =    2 };
   enum { fsSrchThrd_11   =    3 };
   enum { fsSrchThrd_12   =    3 };
   enum { fsSrchThrd_13   =    3 };
   enum { fsSrchThrd_14   =    3 };
   enum { fsSrchThrd_15   =    3 };
   enum { fsSrchThrd_16   =    3 };
   enum { fsSrchThrd_17   =    3 };
   enum { fsSrchThrd_18   =    3 };
   enum { fsSrchThrd_19   =    3 };
   enum { fsSrchThrd_20   =    3 };
   enum { fsSrchThrd_21   =    3 };
   enum { fsSrchThrd_22   =    3 };
   enum { fsSrchThrd_23   =    3 };
   enum { fsSrchThrd_24   =    3 };
   enum { fsSrchThrd_25   =    4 };
   enum { fsSrchThrd_26   =    4 };
   enum { fsSrchThrd_27   =    4 };
   enum { fsSrchThrd_28   =    4 };
   enum { fsSrchThrd_29   =    4 };
   enum { fsSrchThrd_30   =    4 };
   enum { fsSrchThrd_31   =    4 };
   enum { fsSrchThrd_32   =    4 };
   enum { fsSrchThrd_33   =    4 };
   enum { fsSrchThrd_34   =    4 };
   enum { fsSrchThrd_35   =    4 };
   enum { fsSrchThrd_36   =    4 };
   enum { fsSrchThrd_37   =    4 };
   enum { fsSrchThrd_38   =    4 };
   enum { fsSrchThrd_39   =    4 };
   enum { fsSrchThrd_40   =    5 };
   enum { fsSrchThrd_41   =    5 };
   enum { fsSrchThrd_42   =    5 };
   enum { fsSrchThrd_43   =    5 };
   enum { fsSrchThrd_44   =    5 };
   enum { fsSrchThrd_45   =    5 };
   enum { fsSrchThrd_46   =    5 };
   enum { fsSrchThrd_47   =    5 };
   enum { fsSrchThrd_48   =    5 };
   enum { fsSrchThrd_49   =    5 };
   enum { fsSrchThrd_50   =    5 };
   enum { fsSrchThrd_51   =    5 };
   enum { fsSrchThrd_52   =    5 };
   enum { fsSrchThrd_53   =    5 };
   enum { fsSrchThrd_54   =    5 };
   enum { fsSrchThrd_55   =    5 };
   enum { fsSrchThrd_56   =    5 };
   enum { fsSrchThrd_57   =    5 };
   enum { fsSrchThrd_58   =    5 };
   enum { fsSrchThrd_59   =    5 };
   enum { fsSrchThrd_60   =    5 };
   enum { fsSrchThrd_61   =    5 };
   enum { fsSrchThrd_62   =    5 };
   enum { fsSrchThrd_63   =    5 };
   enum { fsSrchThrd_64   =    5 };
   enum { fsSrchThrd_65   =    5 };
   enum { fsSrchThrd_66   =    5 };
   enum { fsSrchThrd_67   =    5 };
   enum { fsSrchThrd_68   =    5 };
   enum { fsSrchThrd_69   =    5 };
   enum { fsSrchThrd_70   =    5 };
   enum { fsSrchThrd_71   =    5 };
   enum { fsSrchThrd_72   =    5 };
   enum { fsSrchThrd_73   =    5 };
   enum { fsSrchThrd_74   =    5 };
   enum { fsSrchThrd_75   =    5 };
   enum { fsSrchThrd_76   =    5 };
};

template<> class config<cuComplex,ARCH_SM13> {
public:
   enum { fsMinDim   =    2 };
   enum { fsMaxDim   =   44 };

   enum { ge2MinBlks =    1 };
   enum { ge2MaxThrds=  768 }; /* sm_13, 20 registers per thread */
   enum { gj1MinBlks =    1 };
   enum { gj1MaxThrds=  768 }; /* sm_13, 20 registers per thread */
   enum { gj2MinBlks =    1 };
   enum { gj2MaxThrds=  768 }; /* sm_13, 20 registers per thread */

   enum { fsDimX_00  =   -1 };
   enum { fsDimX_01  =   -1 };
   enum { fsDimX_02  =    2 };
   enum { fsDimX_03  =    3 };
   enum { fsDimX_04  =    2 };
   enum { fsDimX_05  =    2 };
   enum { fsDimX_06  =    2 };
   enum { fsDimX_07  =    2 };
   enum { fsDimX_08  =    3 };
   enum { fsDimX_09  =    3 };
   enum { fsDimX_10  =    2 };
   enum { fsDimX_11  =    2 };
   enum { fsDimX_12  =    2 };
   enum { fsDimX_13  =    2 };
   enum { fsDimX_14  =    2 };
   enum { fsDimX_15  =    2 };
   enum { fsDimX_16  =    2 };
   enum { fsDimX_17  =    2 };
   enum { fsDimX_18  =    2 };
   enum { fsDimX_19  =    3 };
   enum { fsDimX_20  =    3 };
   enum { fsDimX_21  =    2 };
   enum { fsDimX_22  =    4 };
   enum { fsDimX_23  =    2 };
   enum { fsDimX_24  =    3 };
   enum { fsDimX_25  =    3 };
   enum { fsDimX_26  =    4 };
   enum { fsDimX_27  =    4 };
   enum { fsDimX_28  =    4 };
   enum { fsDimX_29  =    5 };
   enum { fsDimX_30  =    4 };
   enum { fsDimX_31  =    5 };
   enum { fsDimX_32  =    8 };
   enum { fsDimX_33  =    8 };
   enum { fsDimX_34  =    5 };
   enum { fsDimX_35  =    8 };
   enum { fsDimX_36  =    8 };
   enum { fsDimX_37  =    5 };
   enum { fsDimX_38  =    6 };
   enum { fsDimX_39  =    6 };
   enum { fsDimX_40  =    8 };
   enum { fsDimX_41  =    8 };
   enum { fsDimX_42  =    4 };
   enum { fsDimX_43  =    6 };
   enum { fsDimX_44  =    8 };
   enum { fsDimX_45  =   -1 };
   enum { fsDimX_46  =   -1 };
   enum { fsDimX_47  =   -1 };
   enum { fsDimX_48  =   -1 };
   enum { fsDimX_49  =   -1 };
   enum { fsDimX_50  =   -1 };
   enum { fsDimX_51  =   -1 };
   enum { fsDimX_52  =   -1 };
   enum { fsDimX_53  =   -1 };
   enum { fsDimX_54  =   -1 };
   enum { fsDimX_55  =   -1 };
   enum { fsDimX_56  =   -1 };
   enum { fsDimX_57  =   -1 };
   enum { fsDimX_58  =   -1 };
   enum { fsDimX_59  =   -1 };
   enum { fsDimX_60  =   -1 };
   enum { fsDimX_61  =   -1 };
   enum { fsDimX_62  =   -1 };
   enum { fsDimX_63  =   -1 };
   enum { fsDimX_64  =   -1 };
   enum { fsDimX_65  =   -1 };
   enum { fsDimX_66  =   -1 };
   enum { fsDimX_67  =   -1 };
   enum { fsDimX_68  =   -1 };
   enum { fsDimX_69  =   -1 };
   enum { fsDimX_70  =   -1 };
   enum { fsDimX_71  =   -1 };
   enum { fsDimX_72  =   -1 };
   enum { fsDimX_73  =   -1 };
   enum { fsDimX_74  =   -1 };
   enum { fsDimX_75  =   -1 };
   enum { fsDimX_76  =   -1 };

   enum { fsPad_00   =   -1 };
   enum { fsPad_01   =   -1 };
   enum { fsPad_02   =    0 };
   enum { fsPad_03   =    0 };    
   enum { fsPad_04   =    0 };
   enum { fsPad_05   =    0 };
   enum { fsPad_06   =    1 };
   enum { fsPad_07   =    4 };
   enum { fsPad_08   =    3 };
   enum { fsPad_09   =    2 };
   enum { fsPad_10   =    1 };
   enum { fsPad_11   =    2 };
   enum { fsPad_12   =    1 };
   enum { fsPad_13   =    2 };
   enum { fsPad_14   =    1 };
   enum { fsPad_15   =    0 };
   enum { fsPad_16   =    1 };
   enum { fsPad_17   =    0 };
   enum { fsPad_18   =    1 };
   enum { fsPad_19   =    2 };
   enum { fsPad_20   =    1 };
   enum { fsPad_21   =    0 };
   enum { fsPad_22   =    4 };
   enum { fsPad_23   =    2 };
   enum { fsPad_24   =    1 };
   enum { fsPad_25   =    2 };
   enum { fsPad_26   =    4 };
   enum { fsPad_27   =    3 };
   enum { fsPad_28   =    2 };
   enum { fsPad_29   =    0 };
   enum { fsPad_30   =    0 };
   enum { fsPad_31   =    0 };
   enum { fsPad_32   =    4 };
   enum { fsPad_33   =    3 };
   enum { fsPad_34   =    3 };
   enum { fsPad_35   =    1 };
   enum { fsPad_36   =    1 };
   enum { fsPad_37   =    0 };
   enum { fsPad_38   =    0 };
   enum { fsPad_39   =    4 };
   enum { fsPad_40   =    3 };
   enum { fsPad_41   =    3 };
   enum { fsPad_42   =    4 };
   enum { fsPad_43   =    3 };
   enum { fsPad_44   =    1 };
   enum { fsPad_45   =   -1 };
   enum { fsPad_46   =   -1 };
   enum { fsPad_47   =   -1 };
   enum { fsPad_48   =   -1 };
   enum { fsPad_49   =   -1 };
   enum { fsPad_50   =   -1 };
   enum { fsPad_51   =   -1 };
   enum { fsPad_52   =   -1 };
   enum { fsPad_53   =   -1 };
   enum { fsPad_54   =   -1 };
   enum { fsPad_55   =   -1 };
   enum { fsPad_56   =   -1 };
   enum { fsPad_57   =   -1 };
   enum { fsPad_58   =   -1 };
   enum { fsPad_59   =   -1 };
   enum { fsPad_60   =   -1 };
   enum { fsPad_61   =   -1 };
   enum { fsPad_62   =   -1 };
   enum { fsPad_63   =   -1 };
   enum { fsPad_64   =   -1 };
   enum { fsPad_65   =   -1 };
   enum { fsPad_66   =   -1 };
   enum { fsPad_67   =   -1 };
   enum { fsPad_68   =   -1 };
   enum { fsPad_69   =   -1 };
   enum { fsPad_70   =   -1 };
   enum { fsPad_71   =   -1 };
   enum { fsPad_72   =   -1 };
   enum { fsPad_73   =   -1 };
   enum { fsPad_74   =   -1 };
   enum { fsPad_75   =   -1 };
   enum { fsPad_76   =   -1 };

   enum { fsSrchThrd_00   =   -1 };
   enum { fsSrchThrd_01   =   -1 };
   enum { fsSrchThrd_02   =    2 };
   enum { fsSrchThrd_03   =    2 };    
   enum { fsSrchThrd_04   =    2 };
   enum { fsSrchThrd_05   =    2 };
   enum { fsSrchThrd_06   =    2 };
   enum { fsSrchThrd_07   =    2 };
   enum { fsSrchThrd_08   =    2 };
   enum { fsSrchThrd_09   =    2 };
   enum { fsSrchThrd_10   =    2 };
   enum { fsSrchThrd_11   =    2 };
   enum { fsSrchThrd_12   =    2 };
   enum { fsSrchThrd_13   =    2 };
   enum { fsSrchThrd_14   =    2 };
   enum { fsSrchThrd_15   =    2 };
   enum { fsSrchThrd_16   =    2 };
   enum { fsSrchThrd_17   =    2 };
   enum { fsSrchThrd_18   =    2 };
   enum { fsSrchThrd_19   =    2 };
   enum { fsSrchThrd_20   =    3 };
   enum { fsSrchThrd_21   =    3 };
   enum { fsSrchThrd_22   =    3 };
   enum { fsSrchThrd_23   =    3 };
   enum { fsSrchThrd_24   =    3 };
   enum { fsSrchThrd_25   =    3 };
   enum { fsSrchThrd_26   =    3 };
   enum { fsSrchThrd_27   =    3 };
   enum { fsSrchThrd_28   =    3 };
   enum { fsSrchThrd_29   =    3 };
   enum { fsSrchThrd_30   =    3 };
   enum { fsSrchThrd_31   =    3 };
   enum { fsSrchThrd_32   =    4 };
   enum { fsSrchThrd_33   =    4 };
   enum { fsSrchThrd_34   =    4 };
   enum { fsSrchThrd_35   =    4 };
   enum { fsSrchThrd_36   =    4 };
   enum { fsSrchThrd_37   =    4 };
   enum { fsSrchThrd_38   =    4 };
   enum { fsSrchThrd_39   =    4 };
   enum { fsSrchThrd_40   =    4 };
   enum { fsSrchThrd_41   =    4 };
   enum { fsSrchThrd_42   =    4 };
   enum { fsSrchThrd_43   =    4 };
   enum { fsSrchThrd_44   =    4 };
   enum { fsSrchThrd_45   =   -1 };
   enum { fsSrchThrd_46   =   -1 };
   enum { fsSrchThrd_47   =   -1 };
   enum { fsSrchThrd_48   =   -1 };
   enum { fsSrchThrd_49   =   -1 };
   enum { fsSrchThrd_50   =   -1 };
   enum { fsSrchThrd_51   =   -1 };
   enum { fsSrchThrd_52   =   -1 };
   enum { fsSrchThrd_53   =   -1 };
   enum { fsSrchThrd_54   =   -1 };
   enum { fsSrchThrd_55   =   -1 };
   enum { fsSrchThrd_56   =   -1 };
   enum { fsSrchThrd_57   =   -1 };
   enum { fsSrchThrd_58   =   -1 };
   enum { fsSrchThrd_59   =   -1 };
   enum { fsSrchThrd_60   =   -1 };
   enum { fsSrchThrd_61   =   -1 };
   enum { fsSrchThrd_62   =   -1 };
   enum { fsSrchThrd_63   =   -1 };
   enum { fsSrchThrd_64   =   -1 };
   enum { fsSrchThrd_65   =   -1 };
   enum { fsSrchThrd_66   =   -1 };
   enum { fsSrchThrd_67   =   -1 };
   enum { fsSrchThrd_68   =   -1 };
   enum { fsSrchThrd_69   =   -1 };
   enum { fsSrchThrd_70   =   -1 };
   enum { fsSrchThrd_71   =   -1 };
   enum { fsSrchThrd_72   =   -1 };
   enum { fsSrchThrd_73   =   -1 };
   enum { fsSrchThrd_74   =   -1 };
   enum { fsSrchThrd_75   =   -1 };
   enum { fsSrchThrd_76   =   -1 };
};

template<> class config<cuComplex,ARCH_SM20> {
public:
   enum { fsMinDim   =    2 };
   enum { fsMaxDim   =   76 };

   enum { ge2MinBlks =    1 };
   enum { ge2MaxThrds= 1280 }; /* sm_2x, 24 registers per thread */
   enum { gj1MinBlks =    1 };
   enum { gj1MaxThrds= 1536 }; /* sm_2x, 20 registers per thread */
   enum { gj2MinBlks =    1 };
   enum { gj2MaxThrds= 1536 }; /* sm_2x, 20 registers per thread */

   enum { fsDimX_00  =   -1 };
   enum { fsDimX_01  =   -1 };
   enum { fsDimX_02  =    2 };
   enum { fsDimX_03  =    3 };
   enum { fsDimX_04  =    4 };
   enum { fsDimX_05  =    5 };
   enum { fsDimX_06  =    6 };
   enum { fsDimX_07  =    7 };
   enum { fsDimX_08  =    8 };
   enum { fsDimX_09  =    9 };
   enum { fsDimX_10  =    5 };
   enum { fsDimX_11  =    5 };
   enum { fsDimX_12  =    4 };
   enum { fsDimX_13  =    4 };
   enum { fsDimX_14  =    4 };
   enum { fsDimX_15  =    4 };
   enum { fsDimX_16  =    3 };
   enum { fsDimX_17  =    3 };
   enum { fsDimX_18  =    3 };
   enum { fsDimX_19  =    3 };
   enum { fsDimX_20  =    3 };
   enum { fsDimX_21  =    4 };
   enum { fsDimX_22  =    4 };
   enum { fsDimX_23  =    4 };
   enum { fsDimX_24  =    2 };
   enum { fsDimX_25  =    2 };
   enum { fsDimX_26  =    2 };
   enum { fsDimX_27  =    2 };
   enum { fsDimX_28  =    3 };
   enum { fsDimX_29  =    3 };
   enum { fsDimX_30  =    3 };
   enum { fsDimX_31  =    8 }; // adjusted by jojon
   enum { fsDimX_32  =    8 }; // adjusted by jojon
   enum { fsDimX_33  =    8 }; // adjusted by jojon
   enum { fsDimX_34  =    4 };
   enum { fsDimX_35  =    4 };
   enum { fsDimX_36  =    4 };
   enum { fsDimX_37  =    5 };
   enum { fsDimX_38  =    4 };
   enum { fsDimX_39  =    4 };
   enum { fsDimX_40  =    4 };
   enum { fsDimX_41  =    4 };
   enum { fsDimX_42  =    4 };
   enum { fsDimX_43  =    4 };
   enum { fsDimX_44  =    4 };
   enum { fsDimX_45  =    4 };
   enum { fsDimX_46  =    4 };
   enum { fsDimX_47  =    4 };
   enum { fsDimX_48  =    5 };
   enum { fsDimX_49  =    5 };
   enum { fsDimX_50  =    5 };
   enum { fsDimX_51  =    4 };
   enum { fsDimX_52  =    4 };
   enum { fsDimX_53  =    5 };
   enum { fsDimX_54  =    6 };
   enum { fsDimX_55  =    8 };
   enum { fsDimX_56  =    8 };
   enum { fsDimX_57  =    6 };
   enum { fsDimX_58  =    7 };
   enum { fsDimX_59  =    6 };
   enum { fsDimX_60  =    6 };
   enum { fsDimX_61  =    6 };
   enum { fsDimX_62  =    7 };
   enum { fsDimX_63  =    6 };
   enum { fsDimX_64  =    5 };
   enum { fsDimX_65  =    6 };
   enum { fsDimX_66  =    6 };
   enum { fsDimX_67  =    6 };
   enum { fsDimX_68  =    6 };
   enum { fsDimX_69  =    6 };
   enum { fsDimX_70  =    7 };
   enum { fsDimX_71  =    7 };
   enum { fsDimX_72  =    6 };
   enum { fsDimX_73  =    6 };
   enum { fsDimX_74  =    6 };
   enum { fsDimX_75  =    6 };
   enum { fsDimX_76  =    4 };

   enum { fsPad_00   =   -1 };
   enum { fsPad_01   =   -1 };
   enum { fsPad_02   =    0 };
   enum { fsPad_03   =    0 };
   enum { fsPad_04   =    0 };
   enum { fsPad_05   =    0 };
   enum { fsPad_06   =    0 };
   enum { fsPad_07   =    0 };
   enum { fsPad_08   =    0 };
   enum { fsPad_09   =    0 };
   enum { fsPad_10   =    0 };
   enum { fsPad_11   =    1 };
   enum { fsPad_12   =    0 };
   enum { fsPad_13   =    1 };
   enum { fsPad_14   =    0 };
   enum { fsPad_15   =    3 };
   enum { fsPad_16   =    3 };
   enum { fsPad_17   =    2 };
   enum { fsPad_18   =    1 };
   enum { fsPad_19   =    0 };
   enum { fsPad_20   =    2 };
   enum { fsPad_21   =    0 };
   enum { fsPad_22   =    0 };
   enum { fsPad_23   =    0 };
   enum { fsPad_24   =    2 };
   enum { fsPad_25   =    1 };
   enum { fsPad_26   =    0 };
   enum { fsPad_27   =    0 };
   enum { fsPad_28   =    1 };
   enum { fsPad_29   =    1 };
   enum { fsPad_30   =    0 };
   enum { fsPad_31   =    0 };
   enum { fsPad_32   =    3 };
   enum { fsPad_33   =    2 };
   enum { fsPad_34   =    0 };
   enum { fsPad_35   =    1 };
   enum { fsPad_36   =    0 };
   enum { fsPad_37   =    0 };
   enum { fsPad_38   =    1 };
   enum { fsPad_39   =    3 };
   enum { fsPad_40   =    4 };
   enum { fsPad_41   =    3 };
   enum { fsPad_42   =    2 };
   enum { fsPad_43   =    1 };
   enum { fsPad_44   =    0 };
   enum { fsPad_45   =    5 };
   enum { fsPad_46   =    4 };
   enum { fsPad_47   =    3 };
   enum { fsPad_48   =    5 };
   enum { fsPad_49   =    4 };
   enum { fsPad_50   =    3 };
   enum { fsPad_51   =    1 };
   enum { fsPad_52   =    0 };
   enum { fsPad_53   =    0 };
   enum { fsPad_54   =    0 };
   enum { fsPad_55   =    1 };
   enum { fsPad_56   =    0 };
   enum { fsPad_57   =    2 };
   enum { fsPad_58   =    2 };
   enum { fsPad_59   =    1 };
   enum { fsPad_60   =    1 };
   enum { fsPad_61   =    1 };
   enum { fsPad_62   =    1 };
   enum { fsPad_63   =    5 };
   enum { fsPad_64   =    5 };
   enum { fsPad_65   =    5 };
   enum { fsPad_66   =    4 };
   enum { fsPad_67   =    3 };
   enum { fsPad_68   =    2 };
   enum { fsPad_69   =    1 };
   enum { fsPad_70   =    1 };
   enum { fsPad_71   =    0 };
   enum { fsPad_72   =    3 };
   enum { fsPad_73   =    2 };
   enum { fsPad_74   =    1 };
   enum { fsPad_75   =    0 };
   enum { fsPad_76   =    0 };

   enum { fsSrchThrd_00   =   -1 };
   enum { fsSrchThrd_01   =   -1 };
   enum { fsSrchThrd_02   =    2 };
   enum { fsSrchThrd_03   =    2 };    
   enum { fsSrchThrd_04   =    2 };
   enum { fsSrchThrd_05   =    2 };
   enum { fsSrchThrd_06   =    2 };
   enum { fsSrchThrd_07   =    2 };
   enum { fsSrchThrd_08   =    2 };
   enum { fsSrchThrd_09   =    2 };
   enum { fsSrchThrd_10   =    2 };
   enum { fsSrchThrd_11   =    3 };
   enum { fsSrchThrd_12   =    3 };
   enum { fsSrchThrd_13   =    3 };
   enum { fsSrchThrd_14   =    3 };
   enum { fsSrchThrd_15   =    3 };
   enum { fsSrchThrd_16   =    3 };
   enum { fsSrchThrd_17   =    3 };
   enum { fsSrchThrd_18   =    3 };
   enum { fsSrchThrd_19   =    3 };
   enum { fsSrchThrd_20   =    3 };
   enum { fsSrchThrd_21   =    3 };
   enum { fsSrchThrd_22   =    3 };
   enum { fsSrchThrd_23   =    3 };
   enum { fsSrchThrd_24   =    3 };
   enum { fsSrchThrd_25   =    4 };
   enum { fsSrchThrd_26   =    4 };
   enum { fsSrchThrd_27   =    4 };
   enum { fsSrchThrd_28   =    4 };
   enum { fsSrchThrd_29   =    4 };
   enum { fsSrchThrd_30   =    4 };
   enum { fsSrchThrd_31   =    4 };
   enum { fsSrchThrd_32   =    4 };
   enum { fsSrchThrd_33   =    4 };
   enum { fsSrchThrd_34   =    4 };
   enum { fsSrchThrd_35   =    4 };
   enum { fsSrchThrd_36   =    4 };
   enum { fsSrchThrd_37   =    4 };
   enum { fsSrchThrd_38   =    4 };
   enum { fsSrchThrd_39   =    4 };
   enum { fsSrchThrd_40   =    5 };
   enum { fsSrchThrd_41   =    5 };
   enum { fsSrchThrd_42   =    5 };
   enum { fsSrchThrd_43   =    5 };
   enum { fsSrchThrd_44   =    5 };
   enum { fsSrchThrd_45   =    5 };
   enum { fsSrchThrd_46   =    5 };
   enum { fsSrchThrd_47   =    5 };
   enum { fsSrchThrd_48   =    5 };
   enum { fsSrchThrd_49   =    5 };
   enum { fsSrchThrd_50   =    5 };
   enum { fsSrchThrd_51   =    5 };
   enum { fsSrchThrd_52   =    5 };
   enum { fsSrchThrd_53   =    5 };
   enum { fsSrchThrd_54   =    5 };
   enum { fsSrchThrd_55   =    5 };
   enum { fsSrchThrd_56   =    5 };
   enum { fsSrchThrd_57   =    5 };
   enum { fsSrchThrd_58   =    5 };
   enum { fsSrchThrd_59   =    5 };
   enum { fsSrchThrd_60   =    5 };
   enum { fsSrchThrd_61   =    5 };
   enum { fsSrchThrd_62   =    5 };
   enum { fsSrchThrd_63   =    5 };
   enum { fsSrchThrd_64   =    5 };
   enum { fsSrchThrd_65   =    5 };
   enum { fsSrchThrd_66   =    5 };
   enum { fsSrchThrd_67   =    5 };
   enum { fsSrchThrd_68   =    5 };
   enum { fsSrchThrd_69   =    5 };
   enum { fsSrchThrd_70   =    5 };
   enum { fsSrchThrd_71   =    5 };
   enum { fsSrchThrd_72   =    5 };
   enum { fsSrchThrd_73   =    5 };
   enum { fsSrchThrd_74   =    5 };
   enum { fsSrchThrd_75   =    5 };
   enum { fsSrchThrd_76   =    5 };
};

template<> class config<cuDoubleComplex,ARCH_SM13> {
public:
   enum { fsMinDim   =    2 };
   enum { fsMaxDim   =   31 };

   enum { ge2MinBlks =    1 };
   enum { ge2MaxThrds=  640 }; /* sm_13, 24 registers per thread */
   enum { gj1MinBlks =    1 };
   enum { gj1MaxThrds=  640 }; /* sm_13, 24 registers per thread */
   enum { gj2MinBlks =    1 };
   enum { gj2MaxThrds=  640 }; /* sm_13, 24 registers per thread */

   enum { fsDimX_00  =   -1 };
   enum { fsDimX_01  =   -1 };
   enum { fsDimX_02  =    2 };
   enum { fsDimX_03  =    3 };
   enum { fsDimX_04  =    3 };
   enum { fsDimX_05  =    2 };
   enum { fsDimX_06  =    2 };
   enum { fsDimX_07  =    2 };
   enum { fsDimX_08  =    3 };
   enum { fsDimX_09  =    3 };
   enum { fsDimX_10  =    4 };
   enum { fsDimX_11  =    4 };
   enum { fsDimX_12  =    5 };
   enum { fsDimX_13  =    4 };
   enum { fsDimX_14  =    4 };
   enum { fsDimX_15  =    4 };
   enum { fsDimX_16  =    4 };
   enum { fsDimX_17  =    4 };
   enum { fsDimX_18  =    4 };
   enum { fsDimX_19  =    4 };
   enum { fsDimX_20  =    8 };
   enum { fsDimX_21  =    5 };
   enum { fsDimX_22  =    4 };
   enum { fsDimX_23  =    8 };
   enum { fsDimX_24  =    8 };
   enum { fsDimX_25  =    8 };
   enum { fsDimX_26  =    8 };
   enum { fsDimX_27  =    8 };
   enum { fsDimX_28  =    8 };
   enum { fsDimX_29  =    8 };
   enum { fsDimX_30  =    8 };
   enum { fsDimX_31  =    8 };
   enum { fsDimX_32  =   -1 };
   enum { fsDimX_33  =   -1 };
   enum { fsDimX_34  =   -1 };
   enum { fsDimX_35  =   -1 };
   enum { fsDimX_36  =   -1 };
   enum { fsDimX_37  =   -1 };
   enum { fsDimX_38  =   -1 };
   enum { fsDimX_39  =   -1 };
   enum { fsDimX_40  =   -1 };
   enum { fsDimX_41  =   -1 };
   enum { fsDimX_42  =   -1 };
   enum { fsDimX_43  =   -1 };
   enum { fsDimX_44  =   -1 };
   enum { fsDimX_45  =   -1 };
   enum { fsDimX_46  =   -1 };
   enum { fsDimX_47  =   -1 };
   enum { fsDimX_48  =   -1 };
   enum { fsDimX_49  =   -1 };
   enum { fsDimX_50  =   -1 };
   enum { fsDimX_51  =   -1 };
   enum { fsDimX_52  =   -1 };
   enum { fsDimX_53  =   -1 };
   enum { fsDimX_54  =   -1 };
   enum { fsDimX_55  =   -1 };
   enum { fsDimX_56  =   -1 };
   enum { fsDimX_57  =   -1 };
   enum { fsDimX_58  =   -1 };
   enum { fsDimX_59  =   -1 };
   enum { fsDimX_60  =   -1 };
   enum { fsDimX_61  =   -1 };
   enum { fsDimX_62  =   -1 };
   enum { fsDimX_63  =   -1 };
   enum { fsDimX_64  =   -1 };
   enum { fsDimX_65  =   -1 };
   enum { fsDimX_66  =   -1 };
   enum { fsDimX_67  =   -1 };
   enum { fsDimX_68  =   -1 };
   enum { fsDimX_69  =   -1 };
   enum { fsDimX_70  =   -1 };
   enum { fsDimX_71  =   -1 };
   enum { fsDimX_72  =   -1 };
   enum { fsDimX_73  =   -1 };
   enum { fsDimX_74  =   -1 };
   enum { fsDimX_75  =   -1 };
   enum { fsDimX_76  =   -1 };

   enum { fsPad_00   =   -1 };
   enum { fsPad_01   =   -1 };
   enum { fsPad_02   =    1 };
   enum { fsPad_03   =    0 };    
   enum { fsPad_04   =    1 };
   enum { fsPad_05   =    2 };
   enum { fsPad_06   =    1 };
   enum { fsPad_07   =    2 };
   enum { fsPad_08   =    1 };
   enum { fsPad_09   =    2 };
   enum { fsPad_10   =    1 };
   enum { fsPad_11   =    0 };
   enum { fsPad_12   =    1 };
   enum { fsPad_13   =    0 };
   enum { fsPad_14   =    1 };
   enum { fsPad_15   =    0 };
   enum { fsPad_16   =    1 };
   enum { fsPad_17   =    0 };
   enum { fsPad_18   =    1 };
   enum { fsPad_19   =    2 };
   enum { fsPad_20   =    1 };
   enum { fsPad_21   =    0 };
   enum { fsPad_22   =    0 };
   enum { fsPad_23   =    2 };
   enum { fsPad_24   =    2 };
   enum { fsPad_25   =    1 };
   enum { fsPad_26   =    4 };
   enum { fsPad_27   =    3 };
   enum { fsPad_28   =    2 };
   enum { fsPad_29   =    1 };
   enum { fsPad_30   =    1 };
   enum { fsPad_31   =    0 };
   enum { fsPad_32   =   -1 };
   enum { fsPad_33   =   -1 };
   enum { fsPad_34   =   -1 };
   enum { fsPad_35   =   -1 };
   enum { fsPad_36   =   -1 };
   enum { fsPad_37   =   -1 };
   enum { fsPad_38   =   -1 };
   enum { fsPad_39   =   -1 };
   enum { fsPad_40   =   -1 };
   enum { fsPad_41   =   -1 };
   enum { fsPad_42   =   -1 };
   enum { fsPad_43   =   -1 };
   enum { fsPad_44   =   -1 };
   enum { fsPad_45   =   -1 };
   enum { fsPad_46   =   -1 };
   enum { fsPad_47   =   -1 };
   enum { fsPad_48   =   -1 };
   enum { fsPad_49   =   -1 };
   enum { fsPad_50   =   -1 };
   enum { fsPad_51   =   -1 };
   enum { fsPad_52   =   -1 };
   enum { fsPad_53   =   -1 };
   enum { fsPad_54   =   -1 };
   enum { fsPad_55   =   -1 };
   enum { fsPad_56   =   -1 };
   enum { fsPad_57   =   -1 };
   enum { fsPad_58   =   -1 };
   enum { fsPad_59   =   -1 };
   enum { fsPad_60   =   -1 };
   enum { fsPad_61   =   -1 };
   enum { fsPad_62   =   -1 };
   enum { fsPad_63   =   -1 };
   enum { fsPad_64   =   -1 };
   enum { fsPad_65   =   -1 };
   enum { fsPad_66   =   -1 };
   enum { fsPad_67   =   -1 };
   enum { fsPad_68   =   -1 };
   enum { fsPad_69   =   -1 };
   enum { fsPad_70   =   -1 };
   enum { fsPad_71   =   -1 };
   enum { fsPad_72   =   -1 };
   enum { fsPad_73   =   -1 };
   enum { fsPad_74   =   -1 };
   enum { fsPad_75   =   -1 };
   enum { fsPad_76   =   -1 };

   enum { fsSrchThrd_00   =   -1 };
   enum { fsSrchThrd_01   =   -1 };
   enum { fsSrchThrd_02   =    2 };
   enum { fsSrchThrd_03   =    2 };    
   enum { fsSrchThrd_04   =    2 };
   enum { fsSrchThrd_05   =    2 };
   enum { fsSrchThrd_06   =    2 };
   enum { fsSrchThrd_07   =    2 };
   enum { fsSrchThrd_08   =    2 };
   enum { fsSrchThrd_09   =    2 };
   enum { fsSrchThrd_10   =    2 };
   enum { fsSrchThrd_11   =    2 };
   enum { fsSrchThrd_12   =    2 };
   enum { fsSrchThrd_13   =    3 };
   enum { fsSrchThrd_14   =    3 };
   enum { fsSrchThrd_15   =    3 };
   enum { fsSrchThrd_16   =    3 };
   enum { fsSrchThrd_17   =    3 };
   enum { fsSrchThrd_18   =    3 };
   enum { fsSrchThrd_19   =    3 };
   enum { fsSrchThrd_20   =    3 };
   enum { fsSrchThrd_21   =    3 };
   enum { fsSrchThrd_22   =    2 };
   enum { fsSrchThrd_23   =    3 };
   enum { fsSrchThrd_24   =    3 };
   enum { fsSrchThrd_25   =    4 };
   enum { fsSrchThrd_26   =    4 };
   enum { fsSrchThrd_27   =    4 };
   enum { fsSrchThrd_28   =    4 };
   enum { fsSrchThrd_29   =    4 };
   enum { fsSrchThrd_30   =    4 };
   enum { fsSrchThrd_31   =    4 };
   enum { fsSrchThrd_32   =   -1 };
   enum { fsSrchThrd_33   =   -1 };
   enum { fsSrchThrd_34   =   -1 };
   enum { fsSrchThrd_35   =   -1 };
   enum { fsSrchThrd_36   =   -1 };
   enum { fsSrchThrd_37   =   -1 };
   enum { fsSrchThrd_38   =   -1 };
   enum { fsSrchThrd_39   =   -1 };
   enum { fsSrchThrd_40   =   -1 };
   enum { fsSrchThrd_41   =   -1 };
   enum { fsSrchThrd_42   =   -1 };
   enum { fsSrchThrd_43   =   -1 };
   enum { fsSrchThrd_44   =   -1 };
   enum { fsSrchThrd_45   =   -1 };
   enum { fsSrchThrd_46   =   -1 };
   enum { fsSrchThrd_47   =   -1 };
   enum { fsSrchThrd_48   =   -1 };
   enum { fsSrchThrd_49   =   -1 };
   enum { fsSrchThrd_50   =   -1 };
   enum { fsSrchThrd_51   =   -1 };
   enum { fsSrchThrd_52   =   -1 };
   enum { fsSrchThrd_53   =   -1 };
   enum { fsSrchThrd_54   =   -1 };
   enum { fsSrchThrd_55   =   -1 };
   enum { fsSrchThrd_56   =   -1 };
   enum { fsSrchThrd_57   =   -1 };
   enum { fsSrchThrd_58   =   -1 };
   enum { fsSrchThrd_59   =   -1 };
   enum { fsSrchThrd_60   =   -1 };
   enum { fsSrchThrd_61   =   -1 };
   enum { fsSrchThrd_62   =   -1 };
   enum { fsSrchThrd_63   =   -1 };
   enum { fsSrchThrd_64   =   -1 };
   enum { fsSrchThrd_65   =   -1 };
   enum { fsSrchThrd_66   =   -1 };
   enum { fsSrchThrd_67   =   -1 };
   enum { fsSrchThrd_68   =   -1 };
   enum { fsSrchThrd_69   =   -1 };
   enum { fsSrchThrd_70   =   -1 };
   enum { fsSrchThrd_71   =   -1 };
   enum { fsSrchThrd_72   =   -1 };
   enum { fsSrchThrd_73   =   -1 };
   enum { fsSrchThrd_74   =   -1 };
   enum { fsSrchThrd_75   =   -1 };
   enum { fsSrchThrd_76   =   -1 };
};

template<> class config<cuDoubleComplex,ARCH_SM20> {
public:
   enum { fsMinDim   =    2 };
   enum { fsMaxDim   =   53 };

   enum { ge2MinBlks =    1 };
   enum { ge2MaxThrds= 1152 }; /* sm_2x, 28 registers per thread */
   enum { gj1MinBlks =    1 };
   enum { gj1MaxThrds= 1152 }; /* sm_2x, 28 registers per thread */
   enum { gj2MinBlks =    1 };
   enum { gj2MaxThrds= 1152 }; /* sm_2x, 28 registers per thread */

   enum { fsDimX_00  =   -1 };
   enum { fsDimX_01  =   -1 };
   enum { fsDimX_02  =    2 };
   enum { fsDimX_03  =    3 };
   enum { fsDimX_04  =    4 };
   enum { fsDimX_05  =    5 };
   enum { fsDimX_06  =    6 };
   enum { fsDimX_07  =    7 };
   enum { fsDimX_08  =    8 };
   enum { fsDimX_09  =    9 };
   enum { fsDimX_10  =    5 };
   enum { fsDimX_11  =    5 };
   enum { fsDimX_12  =    4 };
   enum { fsDimX_13  =    4 };
   enum { fsDimX_14  =    4 };
   enum { fsDimX_15  =    2 };
   enum { fsDimX_16  =    5 };
   enum { fsDimX_17  =    5 };
   enum { fsDimX_18  =    5 };
   enum { fsDimX_19  =    3 };
   enum { fsDimX_20  =    4 };
   enum { fsDimX_21  =    4 };
   enum { fsDimX_22  =    4 };
   enum { fsDimX_23  =    4 };
   enum { fsDimX_24  =    5 };
   enum { fsDimX_25  =    6 };
   enum { fsDimX_26  =    8 };
   enum { fsDimX_27  =    3 };
   enum { fsDimX_28  =    4 };
   enum { fsDimX_29  =    5 };
   enum { fsDimX_30  =    6 };
   enum { fsDimX_31  =    7 };
   enum { fsDimX_32  =    8 };
   enum { fsDimX_33  =    8 };
   enum { fsDimX_34  =    8 };
   enum { fsDimX_35  =    8 };
   enum { fsDimX_36  =    4 };
   enum { fsDimX_37  =    5 };
   enum { fsDimX_38  =    4 };
   enum { fsDimX_39  =    8 };
   enum { fsDimX_40  =    8 };
   enum { fsDimX_41  =    8 };
   enum { fsDimX_42  =    8 };
   enum { fsDimX_43  =    8 };
   enum { fsDimX_44  =    8 };
   enum { fsDimX_45  =    8 };
   enum { fsDimX_46  =    6 };
   enum { fsDimX_47  =    8 };
   enum { fsDimX_48  =    8 };
   enum { fsDimX_49  =    8 };
   enum { fsDimX_50  =    8 };
   enum { fsDimX_51  =    8 };
   enum { fsDimX_52  =    8 };
   enum { fsDimX_53  =    8 };
   enum { fsDimX_54  =   -1 };
   enum { fsDimX_55  =   -1 };
   enum { fsDimX_56  =   -1 };
   enum { fsDimX_57  =   -1 };
   enum { fsDimX_58  =   -1 };
   enum { fsDimX_59  =   -1 };
   enum { fsDimX_60  =   -1 };
   enum { fsDimX_61  =   -1 };
   enum { fsDimX_62  =   -1 };
   enum { fsDimX_63  =   -1 };
   enum { fsDimX_64  =   -1 };
   enum { fsDimX_65  =   -1 };
   enum { fsDimX_66  =   -1 };
   enum { fsDimX_67  =   -1 };
   enum { fsDimX_68  =   -1 };
   enum { fsDimX_69  =   -1 };
   enum { fsDimX_70  =   -1 };
   enum { fsDimX_71  =   -1 };
   enum { fsDimX_72  =   -1 };
   enum { fsDimX_73  =   -1 };
   enum { fsDimX_74  =   -1 };
   enum { fsDimX_75  =   -1 };
   enum { fsDimX_76  =   -1 };

   enum { fsPad_00   =   -1 };
   enum { fsPad_01   =   -1 };
   enum { fsPad_02   =    1 };
   enum { fsPad_03   =    0 };
   enum { fsPad_04   =    1 };
   enum { fsPad_05   =    0 };
   enum { fsPad_06   =    0 };
   enum { fsPad_07   =    0 };
   enum { fsPad_08   =    1 };
   enum { fsPad_09   =    0 };
   enum { fsPad_10   =    3 };
   enum { fsPad_11   =    2 };
   enum { fsPad_12   =    2 };
   enum { fsPad_13   =    1 };
   enum { fsPad_14   =    4 };
   enum { fsPad_15   =    4 };
   enum { fsPad_16   =    3 };
   enum { fsPad_17   =    4 };
   enum { fsPad_18   =    1 };
   enum { fsPad_19   =    0 };
   enum { fsPad_20   =    0 };
   enum { fsPad_21   =    1 };
   enum { fsPad_22   =    0 };
   enum { fsPad_23   =    2 };
   enum { fsPad_24   =    3 };
   enum { fsPad_25   =    2 };
   enum { fsPad_26   =    2 };
   enum { fsPad_27   =    0 };
   enum { fsPad_28   =    2 };
   enum { fsPad_29   =    0 };
   enum { fsPad_30   =    0 };
   enum { fsPad_31   =    0 };
   enum { fsPad_32   =    4 };
   enum { fsPad_33   =    3 };
   enum { fsPad_34   =    2 };
   enum { fsPad_35   =    1 };
   enum { fsPad_36   =    2 };
   enum { fsPad_37   =    0 };
   enum { fsPad_38   =    0 };
   enum { fsPad_39   =    4 };
   enum { fsPad_40   =    4 };
   enum { fsPad_41   =    3 };
   enum { fsPad_42   =    2 };
   enum { fsPad_43   =    1 };
   enum { fsPad_44   =    2 };
   enum { fsPad_45   =    4 };
   enum { fsPad_46   =    0 };
   enum { fsPad_47   =    4 };
   enum { fsPad_48   =    4 };
   enum { fsPad_49   =    3 };
   enum { fsPad_50   =    2 };
   enum { fsPad_51   =    2 };
   enum { fsPad_52   =    0 };
   enum { fsPad_53   =    2 };
   enum { fsPad_54   =   -1 };
   enum { fsPad_55   =   -1 };
   enum { fsPad_56   =   -1 };
   enum { fsPad_57   =   -1 };
   enum { fsPad_58   =   -1 };
   enum { fsPad_59   =   -1 };
   enum { fsPad_60   =   -1 };
   enum { fsPad_61   =   -1 };
   enum { fsPad_62   =   -1 };
   enum { fsPad_63   =   -1 };
   enum { fsPad_64   =   -1 };
   enum { fsPad_65   =   -1 };
   enum { fsPad_66   =   -1 };
   enum { fsPad_67   =   -1 };
   enum { fsPad_68   =   -1 };
   enum { fsPad_69   =   -1 };
   enum { fsPad_70   =   -1 };
   enum { fsPad_71   =   -1 };
   enum { fsPad_72   =   -1 };
   enum { fsPad_73   =   -1 };
   enum { fsPad_74   =   -1 };
   enum { fsPad_75   =   -1 };
   enum { fsPad_76   =   -1 };

   enum { fsSrchThrd_00   =   -1 };
   enum { fsSrchThrd_01   =   -1 };
   enum { fsSrchThrd_02   =    2 };
   enum { fsSrchThrd_03   =    2 };    
   enum { fsSrchThrd_04   =    2 };
   enum { fsSrchThrd_05   =    2 };
   enum { fsSrchThrd_06   =    2 };
   enum { fsSrchThrd_07   =    2 };
   enum { fsSrchThrd_08   =    2 };
   enum { fsSrchThrd_09   =    2 };
   enum { fsSrchThrd_10   =    2 };
   enum { fsSrchThrd_11   =    3 };
   enum { fsSrchThrd_12   =    3 };
   enum { fsSrchThrd_13   =    3 };
   enum { fsSrchThrd_14   =    3 };
   enum { fsSrchThrd_15   =    3 };
   enum { fsSrchThrd_16   =    3 };
   enum { fsSrchThrd_17   =    3 };
   enum { fsSrchThrd_18   =    3 };
   enum { fsSrchThrd_19   =    3 };
   enum { fsSrchThrd_20   =    3 };
   enum { fsSrchThrd_21   =    3 };
   enum { fsSrchThrd_22   =    3 };
   enum { fsSrchThrd_23   =    4 };
   enum { fsSrchThrd_24   =    4 };
   enum { fsSrchThrd_25   =    4 };
   enum { fsSrchThrd_26   =    4 };
   enum { fsSrchThrd_27   =    4 };
   enum { fsSrchThrd_28   =    4 };
   enum { fsSrchThrd_29   =    4 };
   enum { fsSrchThrd_30   =    4 };
   enum { fsSrchThrd_31   =    4 };
   enum { fsSrchThrd_32   =    5 };
   enum { fsSrchThrd_33   =    5 };
   enum { fsSrchThrd_34   =    5 };
   enum { fsSrchThrd_35   =    5 };
   enum { fsSrchThrd_36   =    5 };
   enum { fsSrchThrd_37   =    5 };
   enum { fsSrchThrd_38   =    5 };
   enum { fsSrchThrd_39   =    5 };
   enum { fsSrchThrd_40   =    5 };
   enum { fsSrchThrd_41   =    5 };
   enum { fsSrchThrd_42   =    5 };
   enum { fsSrchThrd_43   =    5 };
   enum { fsSrchThrd_44   =    6 };
   enum { fsSrchThrd_45   =    6 };
   enum { fsSrchThrd_46   =    6 };
   enum { fsSrchThrd_47   =    6 };
   enum { fsSrchThrd_48   =    6 };
   enum { fsSrchThrd_49   =    6 };
   enum { fsSrchThrd_50   =    6 };
   enum { fsSrchThrd_51   =    6 };
   enum { fsSrchThrd_52   =    6 };
   enum { fsSrchThrd_53   =    6 };
   enum { fsSrchThrd_54   =   -1 };
   enum { fsSrchThrd_55   =   -1 };
   enum { fsSrchThrd_56   =   -1 };
   enum { fsSrchThrd_57   =   -1 };
   enum { fsSrchThrd_58   =   -1 };
   enum { fsSrchThrd_59   =   -1 };
   enum { fsSrchThrd_60   =   -1 };
   enum { fsSrchThrd_61   =   -1 };
   enum { fsSrchThrd_62   =   -1 };
   enum { fsSrchThrd_63   =   -1 };
   enum { fsSrchThrd_64   =   -1 };
   enum { fsSrchThrd_65   =   -1 };
   enum { fsSrchThrd_66   =   -1 };
   enum { fsSrchThrd_67   =   -1 };
   enum { fsSrchThrd_68   =   -1 };
   enum { fsSrchThrd_69   =   -1 };
   enum { fsSrchThrd_70   =   -1 };
   enum { fsSrchThrd_71   =   -1 };
   enum { fsSrchThrd_72   =   -1 };
   enum { fsSrchThrd_73   =   -1 };
   enum { fsSrchThrd_74   =   -1 };
   enum { fsSrchThrd_75   =   -1 };
   enum { fsSrchThrd_76   =   -1 };
};

/* column-major */
#define As(row,col)   As[(n+ofs)*(col)+(row)]

extern __shared__ double2 shmem[];

template<typename T, int matinv, int pad, int arch>
__global__ void 
__launch_bounds__(config<T,arch>::gj1MaxThrds,config<T,arch>::gj1MinBlks)
gauss_jordan_solve_gpu1 (const T *A, T *b, T *x, int n, int batch)
{
   T *As = (T*)shmem;
   double *Val = (double*)(As + (n+pad) * (n+1));
   int *Loc = (int*)(Val + PIVOT_THRDS);
   const int ofs = pad;
   const int tx = threadIdx.x;
   const int ty = threadIdx.y;
   const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;

   if (blkNum >= batch) return;

   if (!matinv) A += blkNum * n * n;
   b += blkNum * n;
   x += blkNum * n;

   /* Load matrix and RHS into shared memory */
   if (ty  < n) As(tx,ty) = A[ty * n + tx];
   if (ty == n) {
      As(tx,ty) = matinv ? ((tx==blkNum)?mkConst<T>(1):mkConst<T>(0)):b[tx];
   }

   int j = 0;
   do {
      /* Look for pivot */
      __syncthreads();
      if ((ty == 0) && (tx < PIVOT_THRDS)) {
         double val0 = absOp (As(j,j));
         int loc0 = j;
         int i = j + 1 + tx;
         T *dp = &As(i,j);
         const int incr = &As(PIVOT_THRDS,0)-&As(0,0);
         while (i < n) {
            double vali = absOp (*dp);
            if (val0 < vali) {
               val0 = vali;
               loc0 = i;
            }
            dp += incr;
            i  += PIVOT_THRDS;
         }
         Loc[tx] = loc0;
         Val[tx] = val0;
      }

      /* Swap current row with pivot */
      __syncthreads();
      if ((tx == j) && (ty >= j)) {
         int Pl = Loc[0];
         if (Val[1] > Val[0]) Pl = Loc[1];
         T tmp = As(Pl,ty);
         As(Pl,ty) = As(tx,ty);
         As(tx,ty) = tmp;
      }

      /* scale current row */
      __syncthreads();
      if ((tx == j) && (ty > j)) {
         As(tx,ty) = mulOp (As(tx,ty), rcpOp (As(tx,j)));
      }

      /* eliminate above and below current row */
      __syncthreads();
      if ((tx != j) && (ty > j)) {
         As(tx,ty) = fmnaOp (As(tx,j), As(j,ty), As(tx,ty));
      }

      j++;
   } while (j < n);

   __syncthreads();
   if (ty == n) x[tx] = As(tx,n);
}

template<typename T, int matinv, int pad, int arch>
__global__ void 
__launch_bounds__(config<T,arch>::gj2MaxThrds,config<T,arch>::gj2MinBlks)
gauss_jordan_solve_gpu2 (const T *A, T *b, T *x, int n, int batch)
{
   T *As = (T*)shmem;
   double *Val = (double*)(As + (n+pad) * (n+1));
   int *Loc= (int*)(Val + PIVOT_THRDS);
   const int ofs = pad;
   const int tx = threadIdx.x;
   const int ty = threadIdx.y;
   const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;

   if (blkNum >= batch) return;

   if (!matinv) A += blkNum * n * n;
   b += blkNum * n;
   x += blkNum * n;

   /* Load matrix and RHS into shared memory */
   for (int i = tx; i < n; i += blockDim.x) {
      if (ty  < n) As(i,ty) = A[ty * n + i];
      if (ty == n) {
         As(i,ty) = matinv ? ((i==blkNum)?mkConst<T>(1):mkConst<T>(0)):b[i];
      }
   }

   int j = 0;
   do {
      /* Look for pivot */
      __syncthreads();
      if ((ty == 0) && (tx < PIVOT_THRDS)) {
         double val0 = absOp (As(j,j));
         int loc0 = j;
         int i = j + 1 + tx;
         T *dp = &As(i,j);
         const int incr = &As(PIVOT_THRDS,0)-&As(0,0);
         while (i < n) {
            double vali = absOp (*dp);
            if (val0 < vali) {
               val0 = vali;
               loc0 = i;
            }
            dp += incr;
            i  += PIVOT_THRDS;
         }
         Loc[tx] = loc0;
         Val[tx] = val0;
      }

      /* Swap current row with pivot */
      __syncthreads();
      if ((tx == 0) && (ty >= j)) {
         int Pl = Loc[0];
         if (Val[1] > Val[0]) Pl = Loc[1];
         T tmp = As(Pl,ty);
         As(Pl,ty) = As(j,ty);
         As(j,ty)  = tmp;
      }

      /* scale current row */
      __syncthreads();
      if ((tx == 0) && (ty > j)) {
         As(j,ty) = mulOp (As(j,ty), rcpOp (As(j,j)));
      }

      /* eliminate above and below current row */
      __syncthreads();
      for (int i = tx; i < n; i += blockDim.x) {
         if ((i != j) && (ty > j)) {
            As(i,ty) = fmnaOp (As(i,j), As(j,ty), As(i,ty));
         }
      }

      j++;
   } while (j < n);

   __syncthreads();
   if ((tx == 0) && (ty < n)) x[ty] = As(ty,n);
}

template<typename T, int matinv, int pad, int pivot_thrds, int arch>
__global__ void 
__launch_bounds__(config<T,arch>::ge2MaxThrds,config<T,arch>::ge2MinBlks)
gauss_solve_gpu2 (const T *A, T *b, T *x, int n, int batch)
{
//   T *As = (T*)shmem;
//   double *Val = (double*)(As + (n+pad) * (n+1));
//   int *Loc = (int*)(Val + pivot_thrds);
//   const int ofs = pad;
//   const int tx = threadIdx.x;
//   const int ty = threadIdx.y;
//   const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;
//
//   if (blkNum >= batch) return;
//
//   if (!matinv) A += blkNum * n * n;
//   b += blkNum * n;
//   x += blkNum * n;
//
//   /* Load matrix and RHS into shared memory */
//   for (int i = tx; i < n; i += blockDim.x) {
//      if (ty  < n) As(i,ty) = A[ty * n + i];
//      if (ty == n) {
//         As(i,ty) = matinv ? ((i==blkNum)?mkConst<T>(1):mkConst<T>(0)):b[i];
//      }
//   }
//
   // Memory load scheme adjusted by jojon 
   T *As = (T*)shmem;
#ifndef MATH_ONLY_SHARED
   double *Val = (double*)(As + (n+pad) * (n+1));
   int *Loc = (int*)(Val + pivot_thrds);
#else
   double _Val = 0;
   int _Loc = 0;
   double *Val = &_Val;
   int    *Loc = &_Loc;
#endif

#ifdef MATH_ONLY_SHARED
   double _tVal = 0;
   int _tLoc = 0;
   double *tVal = &_tVal;
   int *tLoc = &_tLoc;
#endif


   const int ofs = pad;
   const int tx = threadIdx.x;
   const int ty = threadIdx.y;
   const int blkNum = blockIdx.y * gridDim.x + blockIdx.x;

   const int threadId = blockDim.x * threadIdx.y + threadIdx.x;
   const int Nthreads = blockDim.x * blockDim.y;
   const int Nvalues  = n*n;

#if defined(MATH_ONLY_GLOBAL) || defined(MATH_ONLY_SHARED)
   float flag = (float)(int)(ofs/90.0); //Always zero, but the compiler doesn't know that
   const float ft = (float)(int)(ofs/7000.0); //Always zero, but the compiler doesn't know that
   const float f1 = (float)(int)(ofs/8000.0)+1; //Always one, but the compiler doesn't know that
   float ff[4] = {ft,f1,ft,f1};
   T t0 = *((T*)ff);
#endif
#ifdef MATH_ONLY_SHARED
   T *tAs = &t0;
#endif

   if (blkNum >= batch) return;

#ifndef MATH_ONLY_GLOBAL
   if (!matinv)
      A += blkNum * n * n;
   b += blkNum * n;
   x += blkNum * n;
#else
   if( 1.0==n*flag ) {
      if (!matinv)
         A += blkNum * n * n;
      b += blkNum * n;
      x += blkNum * n;
   }
#endif


   /* Load matrix and RHS into shared memory */
   for(int i = threadId; i < Nvalues; i += Nthreads ) {
#ifndef MATH_ONLY_GLOBAL
   #ifndef MATH_ONLY_SHARED
     As(i%n,i/n) = A[i]; //mulOp(A[i], mkConst<T>(2));
   #else
     *tAs = A[i];
   #endif
#else
   #ifndef MATH_ONLY_SHARED
     As(i%n,i/n) = mkConst<T>(f1);
   #else
     *tAs = mkConst<T>(f1);
   #endif
#endif
   }

   for (int i = threadId; i < n; i += Nthreads) {
#ifndef MATH_ONLY_SHARED
      As(i,n) = matinv ? ((i==blkNum)?mkConst<T>(1):mkConst<T>(0)):b[i];
#else
      *tAs = matinv ? ((i==blkNum)?mkConst<T>(1):mkConst<T>(0)):b[i];
#endif
   }

   int j = 0;
   do {
      /* Look for pivot */
      __syncthreads();
      if ((tx == 0) && (ty < pivot_thrds)) {
#ifndef MATH_ONLY_SHARED
         double val0 = absOp (As(j,j));
#else
         double val0 = absOp (*tAs);
#endif
         int loc0 = j;
         int i = j + 1 + ty;
#ifndef MATH_ONLY_SHARED
         T *dp = &As(i,j);
         const int incr = &As(pivot_thrds,0)-&As(0,0);
#else
         T *dp = &tAs[0];
         const int incr = 0;
#endif
         while (i < n) {
            double vali = absOp (*dp);
            if (val0 < vali) {
               val0 = vali;
               loc0 = i;
            }
            dp += incr;
            i  += pivot_thrds;
         }
#ifndef MATH_ONLY_SHARED
         Loc[ty] = loc0;
#else
         if( 1.0==loc0*flag ) {
            Loc[ty] = loc0;
         }
#endif
         if (pivot_thrds > 1) {
#ifndef MATH_ONLY_SHARED
            Val[ty] = val0;
#else
            if( 1.0==loc0*flag ) {
               Val[ty] = val0;
            }
#endif
         }
      }

      /* Swap current row with pivot */
      __syncthreads();
      if ((tx == 0) && (ty >= j)) {
         /* finish pivot reduction */
#ifndef MATH_ONLY_SHARED
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
         T tmp = As(Pl,ty);
         As(Pl,ty)  = As(j,ty);
         As(j,ty)   = tmp;
#else
         int Pl = tLoc[0];
         if (pivot_thrds > 1) {
            double val = tVal[0];
            int i = 1;
            for (; i < (pivot_thrds-1); i++) {
               if (tVal[i%4] > val) {
                  Pl = tLoc[0];
                  val = tVal[0];
               }
            }
            if (tVal[i] > val) {
               Pl = 0;
            }
          }

         if( 1.0==Pl*flag ) {
            T tmp = As(Pl,ty);
            As(Pl,ty)  = As(j,ty);
            As(j,ty)   = tmp;
         }
#endif

      }

      /* scale current row */
      __syncthreads();
      if ((tx == 0) && (ty > j)) {
#ifndef MATH_ONLY_SHARED
         As(j,ty) = mulOp (As(j,ty), rcpOp (As(j,j)));
#else
         tAs[0] = mulOp (tAs[0], tAs[1]);
#endif
      }

      /* eliminate below current row */
      __syncthreads();
      for (int i = j+1+tx; i < n; i += blockDim.x) {
         if (ty > j) {
#ifndef MATH_ONLY_SHARED
            As(i,ty) = fmnaOp (As(i,j), As(j,ty), As(i,ty));
#else
            tAs[0] = fmnaOp (tAs[0], tAs[1], tAs[0]);
#endif
         }
      }

      j++;
   } while (j < n);

   /* As contains unitary upper triangular matrix */
   j = n-1;
   do {
      __syncthreads();
      if ((tx == 0) && (ty < j)) {
#ifndef MATH_ONLY_SHARED
         As(ty,n) = fmnaOp (As(ty,j), As(j,n), As(ty,n));
#else
         tAs[0] = fmnaOp (tAs[0], tAs[1], tAs[2]);
#endif
      }
      j--;
   } while (j);

   __syncthreads();
   if ((tx == 0) && (ty < n)) {
#ifndef MATH_ONLY_GLOBAL
   #ifndef MATH_ONLY_SHARED
      x[ty] = As(ty,n);
   #else
      x[ty] = tAs[0];
   #endif
#else
   #ifndef MATH_ONLY_SHARED
      t0 = As(ty,n);
      if( 1.0==(*((float*)&t0))*flag )
         x[ty] = t0;
   #else
      tAs[1] = tAs[0];
      if( 1.0==(*((float*)&tAs[1]))*flag )
         x[ty] = tAs[1];
   #endif
#endif

   }
}

template <typename T, int arch>
int fast_solve (const T *A_d, T *b_d, T *x_d, int n, int batch, int matinv)
{ 
   typedef void (* func)(const T *A_d, T *b_d, T *x_d, int n, int batch);

   static int padding[77] = {
      config<T,arch>::fsPad_00, config<T,arch>::fsPad_01,
      config<T,arch>::fsPad_02, config<T,arch>::fsPad_03,
      config<T,arch>::fsPad_04, config<T,arch>::fsPad_05,
      config<T,arch>::fsPad_06, config<T,arch>::fsPad_07,
      config<T,arch>::fsPad_08, config<T,arch>::fsPad_09,
      config<T,arch>::fsPad_10, config<T,arch>::fsPad_11,
      config<T,arch>::fsPad_12, config<T,arch>::fsPad_13,
      config<T,arch>::fsPad_14, config<T,arch>::fsPad_15,
      config<T,arch>::fsPad_16, config<T,arch>::fsPad_17,
      config<T,arch>::fsPad_18, config<T,arch>::fsPad_19,
      config<T,arch>::fsPad_20, config<T,arch>::fsPad_21,
      config<T,arch>::fsPad_22, config<T,arch>::fsPad_23,
      config<T,arch>::fsPad_24, config<T,arch>::fsPad_25,
      config<T,arch>::fsPad_26, config<T,arch>::fsPad_27,
      config<T,arch>::fsPad_28, config<T,arch>::fsPad_29,
      config<T,arch>::fsPad_30, config<T,arch>::fsPad_31,
      config<T,arch>::fsPad_32, config<T,arch>::fsPad_33,
      config<T,arch>::fsPad_34, config<T,arch>::fsPad_35,
      config<T,arch>::fsPad_36, config<T,arch>::fsPad_37,
      config<T,arch>::fsPad_38, config<T,arch>::fsPad_39,
      config<T,arch>::fsPad_40, config<T,arch>::fsPad_41,
      config<T,arch>::fsPad_42, config<T,arch>::fsPad_43,
      config<T,arch>::fsPad_44, config<T,arch>::fsPad_45,
      config<T,arch>::fsPad_46, config<T,arch>::fsPad_47,
      config<T,arch>::fsPad_48, config<T,arch>::fsPad_49,
      config<T,arch>::fsPad_50, config<T,arch>::fsPad_51,
      config<T,arch>::fsPad_52, config<T,arch>::fsPad_53,
      config<T,arch>::fsPad_54, config<T,arch>::fsPad_55,
      config<T,arch>::fsPad_56, config<T,arch>::fsPad_57,
      config<T,arch>::fsPad_58, config<T,arch>::fsPad_59,
      config<T,arch>::fsPad_60, config<T,arch>::fsPad_61,
      config<T,arch>::fsPad_62, config<T,arch>::fsPad_63,
      config<T,arch>::fsPad_64, config<T,arch>::fsPad_65,
      config<T,arch>::fsPad_66, config<T,arch>::fsPad_67,
      config<T,arch>::fsPad_68, config<T,arch>::fsPad_69,
      config<T,arch>::fsPad_70, config<T,arch>::fsPad_71,
      config<T,arch>::fsPad_72, config<T,arch>::fsPad_73,
      config<T,arch>::fsPad_74, config<T,arch>::fsPad_75,
      config<T,arch>::fsPad_76
   };

   static func pf[2*77] = {
      0,                                                             /*  0 */
      0,                                                             /*  1 */
      gauss_jordan_solve_gpu1 <T, 0, config<T,arch>::fsPad_02, arch>,/*  2 */
      gauss_jordan_solve_gpu1 <T, 0, config<T,arch>::fsPad_03, arch>,/*  3 */
      gauss_jordan_solve_gpu1 <T, 0, config<T,arch>::fsPad_04, arch>,/*  4 */
      gauss_jordan_solve_gpu1 <T, 0, config<T,arch>::fsPad_05, arch>,/*  5 */
      gauss_jordan_solve_gpu1 <T, 0, config<T,arch>::fsPad_06, arch>,/*  6 */
      gauss_jordan_solve_gpu1 <T, 0, config<T,arch>::fsPad_07, arch>,/*  7 */
      gauss_jordan_solve_gpu1 <T, 0, config<T,arch>::fsPad_08, arch>,/*  8 */
      gauss_jordan_solve_gpu1 <T, 0, config<T,arch>::fsPad_09, arch>,/*  9 */
      gauss_jordan_solve_gpu2 <T, 0, config<T,arch>::fsPad_10, arch>,/* 10 */
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_11, config<T,arch>::fsSrchThrd_11, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_12, config<T,arch>::fsSrchThrd_12, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_13, config<T,arch>::fsSrchThrd_13, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_14, config<T,arch>::fsSrchThrd_14, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_15, config<T,arch>::fsSrchThrd_15, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_16, config<T,arch>::fsSrchThrd_16, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_17, config<T,arch>::fsSrchThrd_17, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_18, config<T,arch>::fsSrchThrd_18, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_19, config<T,arch>::fsSrchThrd_19, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_20, config<T,arch>::fsSrchThrd_20, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_21, config<T,arch>::fsSrchThrd_21, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_22, config<T,arch>::fsSrchThrd_22, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_23, config<T,arch>::fsSrchThrd_23, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_24, config<T,arch>::fsSrchThrd_24, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_25, config<T,arch>::fsSrchThrd_25, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_26, config<T,arch>::fsSrchThrd_26, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_27, config<T,arch>::fsSrchThrd_27, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_28, config<T,arch>::fsSrchThrd_28, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_29, config<T,arch>::fsSrchThrd_29, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_30, config<T,arch>::fsSrchThrd_30, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_31, config<T,arch>::fsSrchThrd_31, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_32, config<T,arch>::fsSrchThrd_32, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_33, config<T,arch>::fsSrchThrd_33, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_34, config<T,arch>::fsSrchThrd_34, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_35, config<T,arch>::fsSrchThrd_35, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_36, config<T,arch>::fsSrchThrd_36, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_37, config<T,arch>::fsSrchThrd_37, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_38, config<T,arch>::fsSrchThrd_38, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_39, config<T,arch>::fsSrchThrd_39, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_40, config<T,arch>::fsSrchThrd_40, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_41, config<T,arch>::fsSrchThrd_41, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_42, config<T,arch>::fsSrchThrd_42, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_43, config<T,arch>::fsSrchThrd_43, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_44, config<T,arch>::fsSrchThrd_44, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_45, config<T,arch>::fsSrchThrd_45, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_46, config<T,arch>::fsSrchThrd_46, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_47, config<T,arch>::fsSrchThrd_47, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_48, config<T,arch>::fsSrchThrd_48, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_49, config<T,arch>::fsSrchThrd_49, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_50, config<T,arch>::fsSrchThrd_50, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_51, config<T,arch>::fsSrchThrd_51, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_52, config<T,arch>::fsSrchThrd_52, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_53, config<T,arch>::fsSrchThrd_53, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_54, config<T,arch>::fsSrchThrd_54, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_55, config<T,arch>::fsSrchThrd_55, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_56, config<T,arch>::fsSrchThrd_56, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_57, config<T,arch>::fsSrchThrd_57, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_58, config<T,arch>::fsSrchThrd_58, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_59, config<T,arch>::fsSrchThrd_59, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_60, config<T,arch>::fsSrchThrd_60, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_61, config<T,arch>::fsSrchThrd_61, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_62, config<T,arch>::fsSrchThrd_62, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_63, config<T,arch>::fsSrchThrd_63, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_64, config<T,arch>::fsSrchThrd_64, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_65, config<T,arch>::fsSrchThrd_65, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_66, config<T,arch>::fsSrchThrd_66, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_67, config<T,arch>::fsSrchThrd_67, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_68, config<T,arch>::fsSrchThrd_68, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_69, config<T,arch>::fsSrchThrd_69, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_70, config<T,arch>::fsSrchThrd_70, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_71, config<T,arch>::fsSrchThrd_71, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_72, config<T,arch>::fsSrchThrd_72, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_73, config<T,arch>::fsSrchThrd_73, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_74, config<T,arch>::fsSrchThrd_74, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_75, config<T,arch>::fsSrchThrd_75, arch>,
      gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_76, config<T,arch>::fsSrchThrd_76, arch>,
      0,                                                             /*  0 */
      0,                                                             /*  1 */
      gauss_jordan_solve_gpu1 <T, 1, config<T,arch>::fsPad_02, arch>,/*  2 */
      gauss_jordan_solve_gpu1 <T, 1, config<T,arch>::fsPad_03, arch>,/*  3 */
      gauss_jordan_solve_gpu1 <T, 1, config<T,arch>::fsPad_04, arch>,/*  4 */
      gauss_jordan_solve_gpu1 <T, 1, config<T,arch>::fsPad_05, arch>,/*  5 */
      gauss_jordan_solve_gpu1 <T, 1, config<T,arch>::fsPad_06, arch>,/*  6 */
      gauss_jordan_solve_gpu1 <T, 1, config<T,arch>::fsPad_07, arch>,/*  7 */
      gauss_jordan_solve_gpu1 <T, 1, config<T,arch>::fsPad_08, arch>,/*  8 */
      gauss_jordan_solve_gpu1 <T, 1, config<T,arch>::fsPad_09, arch>,/*  9 */
      gauss_jordan_solve_gpu2 <T, 1, config<T,arch>::fsPad_10, arch>,/* 10 */
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_11, config<T,arch>::fsSrchThrd_11, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_12, config<T,arch>::fsSrchThrd_12, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_13, config<T,arch>::fsSrchThrd_13, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_14, config<T,arch>::fsSrchThrd_14, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_15, config<T,arch>::fsSrchThrd_15, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_16, config<T,arch>::fsSrchThrd_16, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_17, config<T,arch>::fsSrchThrd_17, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_18, config<T,arch>::fsSrchThrd_18, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_19, config<T,arch>::fsSrchThrd_19, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_20, config<T,arch>::fsSrchThrd_20, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_21, config<T,arch>::fsSrchThrd_21, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_22, config<T,arch>::fsSrchThrd_22, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_23, config<T,arch>::fsSrchThrd_23, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_24, config<T,arch>::fsSrchThrd_24, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_25, config<T,arch>::fsSrchThrd_25, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_26, config<T,arch>::fsSrchThrd_26, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_27, config<T,arch>::fsSrchThrd_27, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_28, config<T,arch>::fsSrchThrd_28, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_29, config<T,arch>::fsSrchThrd_29, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_30, config<T,arch>::fsSrchThrd_30, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_31, config<T,arch>::fsSrchThrd_31, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_32, config<T,arch>::fsSrchThrd_32, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_33, config<T,arch>::fsSrchThrd_33, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_34, config<T,arch>::fsSrchThrd_34, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_35, config<T,arch>::fsSrchThrd_35, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_36, config<T,arch>::fsSrchThrd_36, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_37, config<T,arch>::fsSrchThrd_37, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_38, config<T,arch>::fsSrchThrd_38, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_39, config<T,arch>::fsSrchThrd_39, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_40, config<T,arch>::fsSrchThrd_40, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_41, config<T,arch>::fsSrchThrd_41, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_42, config<T,arch>::fsSrchThrd_42, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_43, config<T,arch>::fsSrchThrd_43, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_44, config<T,arch>::fsSrchThrd_44, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_45, config<T,arch>::fsSrchThrd_45, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_46, config<T,arch>::fsSrchThrd_46, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_47, config<T,arch>::fsSrchThrd_47, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_48, config<T,arch>::fsSrchThrd_48, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_49, config<T,arch>::fsSrchThrd_49, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_50, config<T,arch>::fsSrchThrd_50, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_51, config<T,arch>::fsSrchThrd_51, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_52, config<T,arch>::fsSrchThrd_52, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_53, config<T,arch>::fsSrchThrd_53, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_54, config<T,arch>::fsSrchThrd_54, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_55, config<T,arch>::fsSrchThrd_55, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_56, config<T,arch>::fsSrchThrd_56, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_57, config<T,arch>::fsSrchThrd_57, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_58, config<T,arch>::fsSrchThrd_58, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_59, config<T,arch>::fsSrchThrd_59, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_60, config<T,arch>::fsSrchThrd_60, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_61, config<T,arch>::fsSrchThrd_61, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_62, config<T,arch>::fsSrchThrd_62, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_63, config<T,arch>::fsSrchThrd_63, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_64, config<T,arch>::fsSrchThrd_64, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_65, config<T,arch>::fsSrchThrd_65, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_66, config<T,arch>::fsSrchThrd_66, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_67, config<T,arch>::fsSrchThrd_67, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_68, config<T,arch>::fsSrchThrd_68, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_69, config<T,arch>::fsSrchThrd_69, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_70, config<T,arch>::fsSrchThrd_70, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_71, config<T,arch>::fsSrchThrd_71, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_72, config<T,arch>::fsSrchThrd_72, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_73, config<T,arch>::fsSrchThrd_73, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_74, config<T,arch>::fsSrchThrd_74, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_75, config<T,arch>::fsSrchThrd_75, arch>,
      gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_76, config<T,arch>::fsSrchThrd_76, arch>,
   };

   if (n < config<T,arch>::fsMinDim || n > config<T,arch>::fsMaxDim || 
      batch < 1) {
         return -1;
   }
   static int dimX[77] = { 
      config<T,arch>::fsDimX_00, config<T,arch>::fsDimX_01, 
      config<T,arch>::fsDimX_02, config<T,arch>::fsDimX_03, 
      config<T,arch>::fsDimX_04, config<T,arch>::fsDimX_05, 
      config<T,arch>::fsDimX_06, config<T,arch>::fsDimX_07, 
      config<T,arch>::fsDimX_08, config<T,arch>::fsDimX_09, 
      config<T,arch>::fsDimX_10, config<T,arch>::fsDimX_11, 
      config<T,arch>::fsDimX_12, config<T,arch>::fsDimX_13, 
      config<T,arch>::fsDimX_14, config<T,arch>::fsDimX_15, 
      config<T,arch>::fsDimX_16, config<T,arch>::fsDimX_17, 
      config<T,arch>::fsDimX_18, config<T,arch>::fsDimX_19, 
      config<T,arch>::fsDimX_20, config<T,arch>::fsDimX_21, 
      config<T,arch>::fsDimX_22, config<T,arch>::fsDimX_23, 
      config<T,arch>::fsDimX_24, config<T,arch>::fsDimX_25, 
      config<T,arch>::fsDimX_26, config<T,arch>::fsDimX_27, 
      config<T,arch>::fsDimX_28, config<T,arch>::fsDimX_29, 
      config<T,arch>::fsDimX_30, config<T,arch>::fsDimX_31, 
      config<T,arch>::fsDimX_32, config<T,arch>::fsDimX_33, 
      config<T,arch>::fsDimX_34, config<T,arch>::fsDimX_35, 
      config<T,arch>::fsDimX_36, config<T,arch>::fsDimX_37, 
      config<T,arch>::fsDimX_38, config<T,arch>::fsDimX_39, 
      config<T,arch>::fsDimX_40, config<T,arch>::fsDimX_41, 
      config<T,arch>::fsDimX_42, config<T,arch>::fsDimX_43, 
      config<T,arch>::fsDimX_44, config<T,arch>::fsDimX_45, 
      config<T,arch>::fsDimX_46, config<T,arch>::fsDimX_47, 
      config<T,arch>::fsDimX_48, config<T,arch>::fsDimX_49, 
      config<T,arch>::fsDimX_50, config<T,arch>::fsDimX_51, 
      config<T,arch>::fsDimX_52, config<T,arch>::fsDimX_53, 
      config<T,arch>::fsDimX_54, config<T,arch>::fsDimX_55, 
      config<T,arch>::fsDimX_56, config<T,arch>::fsDimX_57, 
      config<T,arch>::fsDimX_58, config<T,arch>::fsDimX_59,
      config<T,arch>::fsDimX_60, config<T,arch>::fsDimX_61, 
      config<T,arch>::fsDimX_62, config<T,arch>::fsDimX_63, 
      config<T,arch>::fsDimX_64, config<T,arch>::fsDimX_65, 
      config<T,arch>::fsDimX_66, config<T,arch>::fsDimX_67, 
      config<T,arch>::fsDimX_68, config<T,arch>::fsDimX_69,
      config<T,arch>::fsDimX_70, config<T,arch>::fsDimX_71, 
      config<T,arch>::fsDimX_72, config<T,arch>::fsDimX_73, 
      config<T,arch>::fsDimX_74, config<T,arch>::fsDimX_75, 
      config<T,arch>::fsDimX_76
   };
   static int srchThrd[77] = { 
      config<T,arch>::fsSrchThrd_00, config<T,arch>::fsSrchThrd_01,
      config<T,arch>::fsSrchThrd_02, config<T,arch>::fsSrchThrd_03,
      config<T,arch>::fsSrchThrd_04, config<T,arch>::fsSrchThrd_05,
      config<T,arch>::fsSrchThrd_06, config<T,arch>::fsSrchThrd_07,
      config<T,arch>::fsSrchThrd_08, config<T,arch>::fsSrchThrd_09,
      config<T,arch>::fsSrchThrd_10, config<T,arch>::fsSrchThrd_11,  
      config<T,arch>::fsSrchThrd_12, config<T,arch>::fsSrchThrd_13,
      config<T,arch>::fsSrchThrd_14, config<T,arch>::fsSrchThrd_15,
      config<T,arch>::fsSrchThrd_16, config<T,arch>::fsSrchThrd_17,
      config<T,arch>::fsSrchThrd_18, config<T,arch>::fsSrchThrd_19,
      config<T,arch>::fsSrchThrd_20, config<T,arch>::fsSrchThrd_21,
      config<T,arch>::fsSrchThrd_22, config<T,arch>::fsSrchThrd_23,
      config<T,arch>::fsSrchThrd_24, config<T,arch>::fsSrchThrd_25,
      config<T,arch>::fsSrchThrd_26, config<T,arch>::fsSrchThrd_27,
      config<T,arch>::fsSrchThrd_28, config<T,arch>::fsSrchThrd_29,
      config<T,arch>::fsSrchThrd_30, config<T,arch>::fsSrchThrd_31,
      config<T,arch>::fsSrchThrd_32, config<T,arch>::fsSrchThrd_33,
      config<T,arch>::fsSrchThrd_34, config<T,arch>::fsSrchThrd_35,
      config<T,arch>::fsSrchThrd_36, config<T,arch>::fsSrchThrd_37,
      config<T,arch>::fsSrchThrd_38, config<T,arch>::fsSrchThrd_39,
      config<T,arch>::fsSrchThrd_40, config<T,arch>::fsSrchThrd_41,
      config<T,arch>::fsSrchThrd_42, config<T,arch>::fsSrchThrd_43,
      config<T,arch>::fsSrchThrd_44, config<T,arch>::fsSrchThrd_45,
      config<T,arch>::fsSrchThrd_46, config<T,arch>::fsSrchThrd_47,
      config<T,arch>::fsSrchThrd_48, config<T,arch>::fsSrchThrd_49,
      config<T,arch>::fsSrchThrd_50, config<T,arch>::fsSrchThrd_51,
      config<T,arch>::fsSrchThrd_52, config<T,arch>::fsSrchThrd_53,
      config<T,arch>::fsSrchThrd_54, config<T,arch>::fsSrchThrd_55,
      config<T,arch>::fsSrchThrd_56, config<T,arch>::fsSrchThrd_57,
      config<T,arch>::fsSrchThrd_58, config<T,arch>::fsSrchThrd_59,
      config<T,arch>::fsSrchThrd_60, config<T,arch>::fsSrchThrd_61,
      config<T,arch>::fsSrchThrd_62, config<T,arch>::fsSrchThrd_63,
      config<T,arch>::fsSrchThrd_64, config<T,arch>::fsSrchThrd_65,
      config<T,arch>::fsSrchThrd_66, config<T,arch>::fsSrchThrd_67,
      config<T,arch>::fsSrchThrd_68, config<T,arch>::fsSrchThrd_69,
      config<T,arch>::fsSrchThrd_70, config<T,arch>::fsSrchThrd_71,
      config<T,arch>::fsSrchThrd_72, config<T,arch>::fsSrchThrd_73,
      config<T,arch>::fsSrchThrd_74, config<T,arch>::fsSrchThrd_75,
      config<T,arch>::fsSrchThrd_76
   };

   dim3 dimBlock(dimX[n], n+1);
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
   if (arch == ARCH_SM13) {
      pf[4]    = gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_04, config<T,arch>::fsSrchThrd_04, arch>;
      pf[5]    = gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_05, config<T,arch>::fsSrchThrd_05, arch>;
      pf[6]    = gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_06, config<T,arch>::fsSrchThrd_06, arch>;
      pf[7]    = gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_07, config<T,arch>::fsSrchThrd_07, arch>;
      pf[8]    = gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_08, config<T,arch>::fsSrchThrd_08, arch>;
      pf[9]    = gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_09, config<T,arch>::fsSrchThrd_09, arch>;
      pf[10]   = gauss_solve_gpu2 <T, 0, config<T,arch>::fsPad_10, config<T,arch>::fsSrchThrd_10, arch>;
      pf[77+4] = gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_04, config<T,arch>::fsSrchThrd_04, arch>;
      pf[77+5] = gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_05, config<T,arch>::fsSrchThrd_05, arch>;
      pf[77+6] = gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_06, config<T,arch>::fsSrchThrd_06, arch>;
      pf[77+7] = gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_07, config<T,arch>::fsSrchThrd_07, arch>;
      pf[77+8] = gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_08, config<T,arch>::fsSrchThrd_08, arch>;
      pf[77+9] = gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_09, config<T,arch>::fsSrchThrd_09, arch>;
      pf[77+10]= gauss_solve_gpu2 <T, 1, config<T,arch>::fsPad_10, config<T,arch>::fsSrchThrd_10, arch>;
   }

   int smem_size = (sizeof(A_d[0]) * (n + padding[n]) * (n+1) +
      sizeof(T) * srchThrd[n] +
      sizeof(int) * srchThrd[n]);

   pf[77*(!!matinv)+n]<<<dimGrid,dimBlock,smem_size>>>(A_d,b_d,x_d,n,batch);
   cudaError_t err = cudaGetLastError();
   /* Check synchronous errors, i.e. pre-launch */
   if (cudaSuccess != err) {
      return -2;
   }
   return 0;
}

/* C-callable wrapper functions */
// Double precision
int dsolve_batch (double *A, double *b, double *x, int n, int batch)
{
   return fast_solve<double,GPU_ARCH>(A, b, x, n, batch, 0);
}

int zsolve_batch (cuDoubleComplex *A, cuDoubleComplex *b, cuDoubleComplex *x,
                  int n, int batch)
{ 
   return fast_solve<cuDoubleComplex,GPU_ARCH>(A, b, x, n, batch, 0);
}

int dmatinv (double *A, double *Ainv, int n)
{
   return fast_solve<double,GPU_ARCH>(A, 0, Ainv, n, n, 1);
}

int zmatinv (cuDoubleComplex *A, cuDoubleComplex *Ainv, int n)
{
   return fast_solve<cuDoubleComplex,GPU_ARCH>(A, 0, Ainv, n, n, 1);
}

// Single precision
int fsolve_batch (float *A, float *b, float *x, int n, int batch)
{
   return fast_solve<float,GPU_ARCH>(A, b, x, n, batch, 0);
}

int zfsolve_batch (cuComplex *A, cuComplex *b, cuComplex *x,
                   int n, int batch)
{ 
   return fast_solve<cuComplex,GPU_ARCH>(A, b, x, n, batch, 0);
}

int fmatinv (float *A, float *Ainv, int n)
{
   return fast_solve<float,GPU_ARCH>(A, 0, Ainv, n, n, 1);
}

int zfmatinv (cuComplex *A, cuComplex *Ainv, int n)
{
   return fast_solve<cuComplex,GPU_ARCH>(A, 0, Ainv, n, n, 1);
}
