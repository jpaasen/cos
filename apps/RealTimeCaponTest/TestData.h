#pragma once

#include <cuComplex.h>

const int M16 = 16;

cuComplex testRightside[M16] = 
{
   make_cuComplex(1.0,0.0),
   make_cuComplex(1.0,0.0),
   make_cuComplex(1.0,0.0),
   make_cuComplex(1.0,0.0),
   make_cuComplex(1.0,0.0),
   make_cuComplex(1.0,0.0),
   make_cuComplex(1.0,0.0),
   make_cuComplex(1.0,0.0),
   make_cuComplex(1.0,0.0),
   make_cuComplex(1.0,0.0),
   make_cuComplex(1.0,0.0),
   make_cuComplex(1.0,0.0),
   make_cuComplex(1.0,0.0),
   make_cuComplex(1.0,0.0),
   make_cuComplex(1.0,0.0),
   make_cuComplex(1.0,0.0),
};

cuComplex testImag[M16] = 
{
   make_cuComplex(1.0,1.0),
   make_cuComplex(1.0,-1.0),
   make_cuComplex(1.0,1.0),
   make_cuComplex(1.0,-1.0),
   make_cuComplex(1.0,1.0),
   make_cuComplex(1.0,-1.0),
   make_cuComplex(1.0,1.0),
   make_cuComplex(1.0,-1.0),
   make_cuComplex(1.0,1.0),
   make_cuComplex(1.0,-1.0),
   make_cuComplex(1.0,1.0),
   make_cuComplex(1.0,-1.0),
   make_cuComplex(1.0,1.0),
   make_cuComplex(1.0,-1.0),
   make_cuComplex(1.0,1.0),
   make_cuComplex(1.0,-1.0),
};

cuComplex testLeftSide[M16] = 
{
   make_cuComplex(0.000514730453848f,0.0001938351133f),
   make_cuComplex(0.000443358873817f,-0.000149142741101f),
   make_cuComplex(0.000227479661645f,-0.000180262883725f),
   make_cuComplex(5.05981023683e-05f,9.88749634731e-05f),
   make_cuComplex(8.83809192447e-05f,9.29728784204e-05f),
   make_cuComplex(0.00014697268264f,-1.29472493057e-05f),
   make_cuComplex(0.000184322339458f,-5.71845727136e-05f),
   make_cuComplex(0.000118285195332f,8.53223798613e-06f),
   make_cuComplex(0.000300957114955f,-3.82766946289e-05f),
   make_cuComplex(0.000281682940332f,9.96732319888e-05f),
   make_cuComplex(0.000346073355903f,0.000165250251891f),
   make_cuComplex(0.000399347218563f,2.78129167159e-05f),
   make_cuComplex(0.000355561786126f,0.000109284590463f),
   make_cuComplex(0.000549851218965f,6.89198442124e-05f),
   make_cuComplex(0.00062160348564f,-0.000162355611463f),
   make_cuComplex(0.000272466882018f,-0.00026573331826f),
};

cuComplex testX16[M16] = 
{
   make_cuComplex(1.0,0.0),
   make_cuComplex(2.0,0.0),
   make_cuComplex(3.0,0.0),
   make_cuComplex(4.0,0.0),
   make_cuComplex(5.0,0.0),
   make_cuComplex(6.0,0.0),
   make_cuComplex(7.0,0.0),
   make_cuComplex(8.0,0.0),
   make_cuComplex(9.0,0.0),
   make_cuComplex(10.0,0.0),
   make_cuComplex(11.0,0.0),
   make_cuComplex(12.0,0.0),
   make_cuComplex(13.0,0.0),
   make_cuComplex(14.0,0.0),
   make_cuComplex(15.0,0.0),
   make_cuComplex(16.0,0.0),
};

// Resulting R after building (L=8) with testX (Note: This R is singular)
//31.6667   36.6667   41.6667   46.6667   51.6667   56.6667   61.6667   66.6667
//36.6667   42.6667   48.6667   54.6667   60.6667   66.6667   72.6667   78.6667
//41.6667   48.6667   55.6667   62.6667   69.6667   76.6667   83.6667   90.6667
//46.6667   54.6667   62.6667   70.6667   78.6667   86.6667   94.6667  102.6667
//51.6667   60.6667   69.6667   78.6667   87.6667   96.6667  105.6667  114.6667
//56.6667   66.6667   76.6667   86.6667   96.6667  106.6667  116.6667  126.6667
//61.6667   72.6667   83.6667   94.6667  105.6667  116.6667  127.6667  138.6667
//66.6667   78.6667   90.6667  102.6667  114.6667  126.6667  138.6667  150.6667

// Answer before normalization
// 285  330  375  420  465  510  555  600
// 330  384  438  492  546  600  654  708
// 375  438  501  564  627  690  753  816
// 420  492  564  636  708  780  852  924
// 465  546  627  708  789  870  951 1032
// 510  600  690  780  870  960 1050 1140
// 555  654  753  852  951 1050 1149 1248
// 600  708  816  924 1032 1140 1248 1356

// Answer after Yavg = 1 has been performed on X = [testRightSide testX16 testX16] without normalization
//  1.0e+003 *
//
//    1.7982    0.6690    0.7590    0.8490    0.9390    1.0290    1.1190    1.2090
//    0.6690    1.9962    0.8850    0.9930    1.1010    1.2090    1.3170    1.4250
//    0.7590    0.8850    2.2302    1.1370    1.2630    1.3890    1.5150    1.6410
//    0.8490    0.9930    1.1370    2.5002    1.4250    1.5690    1.7130    1.8570
//    0.9390    1.1010    1.2630    1.4250    2.8062    1.7490    1.9110    2.0730
//    1.0290    1.2090    1.3890    1.5690    1.7490    3.1482    2.1090    2.2890
//    1.1190    1.3170    1.5150    1.7130    1.9110    2.1090    3.5262    2.5050
//    1.2090    1.4250    1.6410    1.8570    2.0730    2.2890    2.5050    3.9402

//cuComplex testR[8][8] = 
//{
//	{make_cuComplex(285,0.0),},
//}
const int M32 = 32;
cuComplex testX32[M32] = 
{
   make_cuComplex(1.0,0.0),
   make_cuComplex(2.0,0.0),
   make_cuComplex(3.0,0.0),
   make_cuComplex(4.0,0.0),
   make_cuComplex(5.0,0.0),
   make_cuComplex(6.0,0.0),
   make_cuComplex(7.0,0.0),
   make_cuComplex(8.0,0.0),
   make_cuComplex(9.0,0.0),
   make_cuComplex(10.0,0.0),
   make_cuComplex(11.0,0.0),
   make_cuComplex(12.0,0.0),
   make_cuComplex(13.0,0.0),
   make_cuComplex(14.0,0.0),
   make_cuComplex(15.0,0.0),
   make_cuComplex(16.0,0.0),
   make_cuComplex(17.0,0.0),
   make_cuComplex(18.0,0.0),
   make_cuComplex(19.0,0.0),
   make_cuComplex(20.0,0.0),
   make_cuComplex(21.0,0.0),
   make_cuComplex(22.0,0.0),
   make_cuComplex(23.0,0.0),
   make_cuComplex(24.0,0.0),
   make_cuComplex(25.0,0.0),
   make_cuComplex(26.0,0.0),
   make_cuComplex(27.0,0.0),
   make_cuComplex(28.0,0.0),
   make_cuComplex(29.0,0.0),
   make_cuComplex(30.0,0.0),
   make_cuComplex(31.0,0.0),
   make_cuComplex(32.0,0.0),
   /*make_cuComplex(1.0,0.0),
   make_cuComplex(2.0,0.0),
   make_cuComplex(3.0,0.0),
   make_cuComplex(4.0,0.0),
   make_cuComplex(5.0,0.0),
   make_cuComplex(6.0,0.0),
   make_cuComplex(7.0,0.0),
   make_cuComplex(8.0,0.0),
   make_cuComplex(9.0,0.0),
   make_cuComplex(10.0,0.0),
   make_cuComplex(11.0,0.0),
   make_cuComplex(12.0,0.0),
   make_cuComplex(13.0,0.0),
   make_cuComplex(14.0,0.0),
   make_cuComplex(15.0,0.0),
   make_cuComplex(16.0,0.0),*/
};

// Resulting R after building (L=16) with testX (Note: This R is singular)
//        1497        1378        1275        1188        1117        1062        1023        1000         993        1002        1027        1068        1125        1198        1287        1392
//        1378        1500        1382        1280        1194        1124        1070        1032        1010        1004        1014        1040        1082        1140        1214        1304
//        1275        1382        1505        1388        1287        1202        1133        1080        1043        1022        1017        1028        1055        1098        1157        1232
//        1188        1280        1388        1512        1396        1296        1212        1144        1092        1056        1036        1032        1044        1072        1116        1176
//        1117        1194        1287        1396        1521        1406        1307        1224        1157        1106        1071        1052        1049        1062        1091        1136
//        1062        1124        1202        1296        1406        1532        1418        1320        1238        1172        1122        1088        1070        1068        1082        1112
//        1023        1070        1133        1212        1307        1418        1545        1432        1335        1254        1189        1140        1107        1090        1089        1104
//        1000        1032        1080        1144        1224        1320        1432        1560        1448        1352        1272        1208        1160        1128        1112        1112
//         993        1010        1043        1092        1157        1238        1335        1448        1577        1466        1371        1292        1229        1182        1151        1136
//        1002        1004        1022        1056        1106        1172        1254        1352        1466        1596        1486        1392        1314        1252        1206        1176
//        1027        1014        1017        1036        1071        1122        1189        1272        1371        1486        1617        1508        1415        1338        1277        1232
//        1068        1040        1028        1032        1052        1088        1140        1208        1292        1392        1508        1640        1532        1440        1364        1304
//        1125        1082        1055        1044        1049        1070        1107        1160        1229        1314        1415        1532        1665        1558        1467        1392
//        1198        1140        1098        1072        1062        1068        1090        1128        1182        1252        1338        1440        1558        1692        1586        1496
//        1287        1214        1157        1116        1091        1082        1089        1112        1151        1206        1277        1364        1467        1586        1721        1616
//        1392        1304        1232        1176        1136        1112        1104        1112        1136        1176        1232        1304        1392        1496        1616        1752
