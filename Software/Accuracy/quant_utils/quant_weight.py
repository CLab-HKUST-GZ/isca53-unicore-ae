import torch
import torch.nn as nn
from typing import Optional
import time
import gc

UNICORE_DATATYPE = "mixed_unicore"
UNICORE_AUTO_PREFIX = "mixed_unicore_auto_b"


#################################  3-bit Datatypes  #################################
INT3 = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
FP3 = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
FP3_ER_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
FP3_ER_NEG = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]
FP3_EA_POS = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0]
FP3_EA_NEG = [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]

#------------------UniCore 3-bit------------------#
FP3_E1M1_4 = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0          ]    # No.0
FP3_E1M1_5 = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0,      5.0     ]    # No.2.3
FP3_E1M1_6 = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0,           6.0]    # No.4

FP3_E2M0_ER_6  = [-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0          ]
FP3_E2M0_10 = [-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0, 10.0          ]
FP3_E2M0_12 = [-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0,      12.0     ]
FP3_E2M0_14 = [-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0,           14.0]

FP3_E2M0_NZ =     [-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0]
#------------------UniCore 3-bit------------------#

#################################  4-bit Datatypes  #################################
INT4 = [-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
FLINT4 = [-16.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 16.0]
FP4_E2M1 = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_ER_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
FP4_ER_NEG = [-12.0, -10.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
FP4_EA_POS = [-12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
FP4_EA_NEG = [-16.0, -12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]

FP4_E1M2 = [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
FP4_E3M0 = [-16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

#------------------MANT 4-bit------------------#
# MANT = A * INT + 2 ** INT
MANT4_A0 = [-128.0, -64.0, -32.0, -16.0, -8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]
MANT4_A5 = [-163.0, -94.0, -57.0, -36.0, -23.0, -14.0, -7.0, -1.0, 1.0, 7.0, 14.0, 23.0, 36.0, 57.0, 94.0, 163.0]
MANT4_A10 = [-198.0, -124.0, -82.0, -56.0, -38.0, -24.0, -12.0, -1.0, 1.0, 12.0, 24.0, 38.0, 56.0, 82.0, 124.0, 198.0]
MANT4_A17 = [-247.0, -166.0, -117.0, -84.0, -59.0, -38.0, -19.0, -1.0, 1.0, 19.0, 38.0, 59.0, 84.0, 117.0, 166.0, 247.0]
MANT4_A20 = [-268.0, -184.0, -132.0, -96.0, -68.0, -44.0, -22.0, -1.0, 1.0, 22.0, 44.0, 68.0, 96.0, 132.0, 184.0, 268.0]
MANT4_A30 = [-338.0, -244.0, -182.0, -136.0, -98.0, -64.0, -32.0, -1.0, 1.0, 32.0, 64.0, 98.0, 136.0, 182.0, 244.0, 338.0]
MANT4_A40 = [-408.0, -304.0, -232.0, -176.0, -128.0, -84.0, -42.0, -1.0, 1.0, 42.0, 84.0, 128.0, 176.0, 232.0, 304.0, 408.0]
MANT4_A50 = [-478.0, -364.0, -282.0, -216.0, -158.0, -104.0, -52.0, -1.0, 1.0, 52.0, 104.0, 158.0, 216.0, 282.0, 364.0, 478.0]
MANT4_A60 = [-548.0, -424.0, -332.0, -256.0, -188.0, -124.0, -62.0, -1.0, 1.0, 62.0, 124.0, 188.0, 256.0, 332.0, 424.0, 548.0]
MANT4_A70 = [-678.0, -484.0, -382.0, -296.0, -218.0, -144.0, -72.0, -1.0, 1.0, 72.0, 144.0, 218.0, 296.0, 382.0, 484.0, 678.0]
MANT4_A80 = [-688.0, -544.0, -432.0, -336.0, -248.0, -164.0, -82.0, -1.0, 1.0, 82.0, 164.0, 248.0, 336.0, 432.0, 544.0, 688.0]
MANT4_A90 = [-758.0, -604.0, -482.0, -376.0, -278.0, -184.0, -92.0, -1.0, 1.0, 92.0, 184.0, 278.0, 376.0, 482.0, 604.0, 758.0]
MANT4_A100 = [-828.0, -664.0, -532.0, -416.0, -308.0, -204.0, -102.0, -1.0, 1.0, 102.0, 204.0, 308.0, 416.0, 532.0, 664.0, 828.0]
MANT4_A110 = [-898.0, -724.9, -582.0, -456.0, -338.0, -224.0, -112.0, -1.0, 1.0, 112.0, 224.0, 338.0, 456.0, 582.0, 724.9, 898.0]
MANT4_A120 = [-968.0, -784.0, -632.0, -496.0, -368.0, -244.0, -122.0, -1.0, 1.0, 122.0, 244.0, 368.0, 496.0, 632.0, 784.0, 968.0]

#------------------MANT 4-bit------------------#

#------------------UniCore 4-bit------------------#
A = [0.25, 0.3125,]

FP4_E1M2_ER_025_POS = [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
FP4_E1M2_ER_025_NEG = [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
FP4_E1M2_ER_075_POS = [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
FP4_E1M2_ER_075_NEG = [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.75, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
FP4_E1M2_ER_125_POS = [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5]
FP4_E1M2_ER_125_NEG = [-3.5, -3.0, -2.5, -2.0, -1.5, -1.25, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

FP4_E1M2_4_POS  = [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 ]
FP4_E1M2_4_NEG  = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 ]
FP4_E1M2_5_POS  = [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 5.0 ]
FP4_E1M2_5_NEG  = [-5.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 ]
FP4_E1M2_6_POS  = [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 6.0 ]
FP4_E1M2_6_NEG  = [-6.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 ]
FP4_E1M2_7_POS  = [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 7.0 ]
FP4_E1M2_7_NEG  = [-7.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 ]

FP4_E2M1_ER_025_POS  = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 ]
FP4_E2M1_ER_025_NEG  = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 ]
FP4_E2M1_ER_075_POS  = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 ]
FP4_E2M1_ER_075_NEG  = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.75, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 ]
FP4_E2M1_ER_250_POS  = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0 ]
FP4_E2M1_ER_250_NEG  = [-6.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 ]
FP4_E2M1_ER_500_POS  = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0 ]
FP4_E2M1_ER_500_NEG  = [-6.0, -5.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 ]

FP4_E2M1_7_POS  = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 7.0 ]
FP4_E2M1_7_NEG  = [-7.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 ]
FP4_E2M1_8_POS  = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0 ]
FP4_E2M1_8_NEG  = [-8.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 ]
FP4_E2M1_10_POS = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0]
FP4_E2M1_10_NEG = [-10.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_E2M1_12_POS = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 12.0]
FP4_E2M1_12_NEG = [-12.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_E2M1_14_POS = [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 14.0]
FP4_E2M1_14_NEG = [-14.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

FP4_E3M0_NZ = [-16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.25, 0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
#------------------UniCore 4-bit------------------#

#################################  5-bit Datatypes  #################################
INT5 = [-15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
FLINT5 = [-64.0, -32.0, -24.0, -16.0, -14.0, -12.0, -10.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 24.0, 32.0, 64.0]
FP5_E2M2 = [-28.0, -24.0, -20.0, -16.0, -14.0, -12.0, -10.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0]
FP5_E3M1 = [-192.0, -128.0, -96.0, -64.0, -48.0, -32.0, -24.0, -16.0, -12.0, -8.0, -6.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0, 192.0]

#################################  6-bit Datatypes  #################################
INT6 = [
    -31.0, -30.0, -29.0, -28.0, -27.0, -26.0, -25.0, -24.0, -23.0, -22.0, -21.0, -20.0, -19.0, -18.0, -17.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0
]
FP6_E2M3 = [
    -60.0, -56.0, -52.0, -48.0, -44.0, -40.0, -36.0, -32.0, -30.0, -28.0, -26.0, -24.0, -22.0, -20.0, -18.0, -16.0, -15.0, -14.0, -13.0, -12.0, -11.0, -10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0
]
FP6_E3M2 = [
    -448.0, -384.0, -320.0, -256.0, -224.0, -192.0, -160.0, -128.0, -112.0, -96.0, -80.0, -64.0, -56.0, -48.0, -40.0, -32.0, -28.0, -24.0, -20.0, -16.0, -14.0, -12.0, -10.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0,
    0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0, 28.0, 32.0, 40.0, 48.0, 56.0, 64.0, 80.0, 96.0, 112.0, 128.0, 160.0, 192.0, 224.0, 256.0, 320.0, 384.0, 448.0
]

DATATYPE_MAPPING_3_BIT = {
    'int3': INT3, 'fp3': FP3, 
    'fp3_er_pos': FP3_ER_POS, 'fp3_er_neg': FP3_ER_NEG, 
    'fp3_ea_pos': FP3_EA_POS, 'fp3_ea_neg': FP3_EA_NEG, 
    'fp3_e1m1_4': FP3_E1M1_4, 'fp3_e1m1_5': FP3_E1M1_5, 'fp3_e1m1_6': FP3_E1M1_6,
    'fp3_e2m0_er_6': FP3_E2M0_ER_6, 'fp3_e2m0_10': FP3_E2M0_10, 'fp3_e2m0_12': FP3_E2M0_12, 'fp3_e2m0_14': FP3_E2M0_14,
    'fp3_e2m0_nz': FP3_E2M0_NZ
}
DATATYPE_MAPPING_3_BIT_MX = {
    'mx_int3': INT3, 'mx_fp3': FP3
}
DATATYPE_MAPPING_3_BIT_NVFP = {
    'nvfp3': FP3
}

DATATYPE_MAPPING_4_BIT = {
    'int4': INT4, 'fp4': FP4_E2M1, 'flint4': FLINT4,
    'fp4_er_pos': FP4_ER_POS, 'fp4_er_neg': FP4_ER_NEG, 
    'fp4_ea_pos': FP4_EA_POS, 'fp4_ea_neg': FP4_EA_NEG, 
    'fp4_e1m2': FP4_E1M2, 'fp4_e3m0': FP4_E3M0,
    
    'mant4_a0': MANT4_A0,
    'mant4_a5': MANT4_A5,
    'mant4_a10': MANT4_A10,
    'mant4_a17': MANT4_A17,
    'mant4_a20': MANT4_A20,
    'mant4_a30': MANT4_A30,
    'mant4_a40': MANT4_A40,
    'mant4_a50': MANT4_A50,
    'mant4_a60': MANT4_A60,
    'mant4_a70': MANT4_A70,
    'mant4_a80': MANT4_A80,
    'mant4_a90': MANT4_A90,
    'mant4_a100': MANT4_A100,
    'mant4_a110': MANT4_A110,
    'mant4_a120': MANT4_A120,
    
    'fp4_e1m2_4_pos': FP4_E1M2_4_POS, 'fp4_e1m2_4_neg': FP4_E1M2_4_NEG,
    'fp4_e1m2_5_pos': FP4_E1M2_5_POS, 'fp4_e1m2_5_neg': FP4_E1M2_5_NEG,
    'fp4_e1m2_6_pos': FP4_E1M2_6_POS, 'fp4_e1m2_6_neg': FP4_E1M2_6_NEG,
    'fp4_e1m2_7_pos': FP4_E1M2_7_POS, 'fp4_e1m2_7_neg': FP4_E1M2_7_NEG,
    'fp4_e2m1_7_pos': FP4_E2M1_7_POS, 'fp4_e2m1_7_neg': FP4_E2M1_7_NEG,
    'fp4_e2m1_8_pos': FP4_E2M1_8_POS, 'fp4_e2m1_8_neg': FP4_E2M1_8_NEG,
    'fp4_e2m1_10_pos': FP4_E2M1_10_POS, 'fp4_e2m1_10_neg': FP4_E2M1_10_NEG,
    'fp4_e2m1_12_pos': FP4_E2M1_12_POS, 'fp4_e2m1_12_neg': FP4_E2M1_12_NEG,
    'fp4_e2m1_14_pos': FP4_E2M1_14_POS, 'fp4_e2m1_14_neg': FP4_E2M1_14_NEG,
    'fp4_e3m0_nz': FP4_E3M0_NZ,
    'fp4_e1m2_er_025_pos': FP4_E1M2_ER_025_POS, 'fp4_e1m2_er_025_neg': FP4_E1M2_ER_025_NEG,
    'fp4_e1m2_er_075_pos': FP4_E1M2_ER_075_POS, 'fp4_e1m2_er_075_neg': FP4_E1M2_ER_075_NEG,
    'fp4_e1m2_er_125_pos': FP4_E1M2_ER_125_POS, 'fp4_e1m2_er_125_neg': FP4_E1M2_ER_125_NEG,
    'fp4_e2m1_er_025_pos': FP4_E2M1_ER_025_POS, 'fp4_e2m1_er_025_neg': FP4_E2M1_ER_025_NEG,
    'fp4_e2m1_er_075_pos': FP4_E2M1_ER_075_POS, 'fp4_e2m1_er_075_neg': FP4_E2M1_ER_075_NEG,
    'fp4_e2m1_er_250_pos': FP4_E2M1_ER_250_POS, 'fp4_e2m1_er_250_neg': FP4_E2M1_ER_250_NEG,
    'fp4_e2m1_er_500_pos': FP4_E2M1_ER_500_POS, 'fp4_e2m1_er_500_neg': FP4_E2M1_ER_500_NEG,
}
DATATYPE_MAPPING_4_BIT_MX = {
    'mx_int4': INT4, 'mxfp4': FP4_E2M1 
}
DATATYPE_MAPPING_4_BIT_NVFP = {
    'nvfp4': FP4_E2M1 
}

DATATYPE_MAPPING_5_BIT = {
    'int5': INT5, 'fp5': FP5_E2M2, 'flint5': FLINT5,
    'fp5_e2m2': FP5_E2M2, 'fp5_e3m1': FP5_E3M1
}

DATATYPE_MAPPING_6_BIT = {
    'int6': INT6, 'fp6': FP6_E2M3, 
    'fp6_e2m3': FP6_E2M3, 'fp6_e3m2': FP6_E3M2
}

@torch.no_grad()
def quant_int(w_fp16, wq_bits:int=4, group_size: Optional[int]=None):
    """
        Symmetric INT quantization.
    """    
    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16
    else:
        K, C = w_fp16.size() # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size)
    
    rmax = torch.amax(w_fp16_new.abs(), dim=-1, keepdim=True)
    qmax = 2 ** (wq_bits - 1) - 1
    qmin = -qmax
    scale_fp = rmax / qmax
    scale_fp = scale_fp.clamp(min=1e-5, max=1e4)
    q_tensor = torch.clamp(torch.round(w_fp16_new / scale_fp), min=qmin, max=qmax)

    w_fp16_new = q_tensor * scale_fp
    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)


# ---------------- Dynamic UniCore helpers (mixed_unicore_auto) ---------------- #
@torch.no_grad()
def _unicore_base_codebook(family: str):
    """
    Return the 15-level symmetric base codebook (without the extra '-0' remap slot) for a given family.
    family in { 'e1m2', 'e2m1' }
    """
    if family == 'e1m2':
        # 15 values, symmetric around 0, max abs 3.5
        return [-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5,
                 0.0,
                 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    elif family == 'e2m1':
        return [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
                 0.0,
                 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    elif family == 'e3m0':
        return [-16.0, -8.0, -4.0, -2.0, -1.0, -0.5, -0.25,
                 0.0,
                 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    elif family == 'e1m2_i':
        return [-7., -6., -5., -4., -1.5, -1.0, -0.5,
                 0.0,
                 0.5, 1.0, 1.5, 4.0, 5.0, 6.0, 7.0]
    elif family == 'e1m1':
        return [-3., -2., -1.,
                 0.0,
                 1., 2., 3.]
    elif family == 'e2m0':
        return [-4., -2., -1., 
                0.,
                1., 2., 4.]
    elif family == 'e1m1_i':
        return [-6., -4., -1.,
                0.0,
                1., 4., 6.]
    elif family == 'e2m0_i':
        return [-16., -8., -1.,
                0.0,
                1., 8., 16.]
    else:
        raise ValueError(f"Unsupported UniCore family '{family}'")


@torch.no_grad()
def _build_unicore_codebook(family: str, side: str, a_value: float):
    """
    Build a 15(+1) level codebook for UniCore given a base family, a side ('pos' or 'neg'), and a candidate 'a'.
    The extra level corresponds to the '-0' remap value.
    """
    base = _unicore_base_codebook(family)
    if side not in ('pos', 'neg'):
        raise ValueError(f"side must be 'pos' or 'neg', got {side}")

    # Insert the extra level according to side
    if side == 'pos':
        extended = base + [float(abs(a_value))]
    else:
        extended = base + [-float(abs(a_value))]

    # Ensure strictly sorted values (remove duplicates if any)
    extended = sorted(set(extended))
    return extended


@torch.no_grad()
def _generate_a_candidates(E: int = 4, M: int = 2, bias: int = 3):
    """
    Generate candidate 'a' values following the grid shown in search_a.py:
        a = 2**(e - bias) * (1 + m / 2**M)
        with e in [1, 2**(E-1)-1] and m in [0, 2**M-1]
    Returns a sorted list of unique positive 'a' values.
    """
    # By default follow search_a.py where E=4, M=2, bias=3
    a_vals = set()
    for e in range(1, 2 ** (E - 1)):
        for m in range(0, 2 ** M):
            a = 2. ** (e - bias) * (1 + m / (2 ** M))
            if a > 0:
                a_vals.add(float(a))
    a_vals.add(0.)
    a_list = sorted(a_vals)
    a_list = [a for a in a_list if not (0.25 < a < 0.5)]
    return a_list


@torch.no_grad()
def _quant_with_codebook(w_fp16, allow_value, group_size: Optional[int] = None, sign_scale: bool = False, clip_search: bool = False, clip_grid: int = 100, clip_maxshrink: float = 0.8):
    """
    Generic quantizer that uses a provided sorted 'allow_value' codebook (list of floats).
    Mirrors quant_datatype() behavior but without relying on DATATYPE_MAPPING.
    """
    # Prepare codebook (as tensor) and mid-points for binning
    allow_value = sorted(allow_value)

    # Grouping
    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.bfloat16)
    else:
        K, C = w_fp16.size()  # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.bfloat16)

    vals = torch.tensor(allow_value, device=w_fp16_new.device, dtype=w_fp16_new.dtype)
    mids = (vals[:-1] + vals[1:]) * 0.5

    # Determine scale per group
    if sign_scale:
        vmin, vmax = torch.aminmax(w_fp16_new, dim=-1, keepdim=True)
        abs_vmax = vmax.abs()
        abs_vmin = vmin.abs()
        rmax = torch.where(abs_vmax >= abs_vmin, vmax, vmin)
    else:
        rmax = torch.amax(w_fp16_new.abs(), dim=-1, keepdim=True)

    qmax = max([abs(x) for x in allow_value])
    scale_fp = rmax / qmax

    # Optional: quantize the scale to a tiny float (E=1, M=6) like in quant_datatype()
    scale_fp_max = torch.amax(scale_fp.abs(), keepdim=True)
    Mq = 6.0
    Eq = 1.0
    bias_q = 2 ** (Eq - 1) - 1
    max_float = (2 - 2 ** (-Mq)) * 2 ** (2 ** Eq - 1 - bias_q)
    min_float = -max_float
    scale_fp_scale = scale_fp_max / max_float
    scale_fp_unscaled = (scale_fp / scale_fp_scale)
    scale_fp_unscaled = torch.clamp(scale_fp_unscaled, min_float, max_float)
    tensor_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(scale_fp_unscaled)) + bias_q)).detach(), 1.0)
    scales = 2.0 ** (tensor_log_scales - Mq - bias_q)
    scale_fp_unscaled = (scale_fp_unscaled / scales).round()
    scale_fp_unscaled = scale_fp_unscaled * scales
    scale_fp = scale_fp_unscaled * scale_fp_scale

    # Quantize with nearest bins defined by mid_value
    x = w_fp16_new / scale_fp
    is_negative_scale = sign_scale & (scale_fp < 0)
    idx_pos = torch.bucketize(x, mids, right=False)
    idx_neg = torch.bucketize(x, mids, right=True)
    idx = torch.where(is_negative_scale, idx_neg, idx_pos)
    q_tensor = vals[idx]

    w_q = q_tensor * scale_fp
    
    if clip_search:
        w_base = w_fp16_new
        q_best = q_tensor
        scale_best = scale_fp
        best_err = (q_tensor - x).to(torch.float32).pow(2).sum(-1)
        num_steps = int(max(1, int(clip_maxshrink * clip_grid)))
        for j in range(num_steps):
            shrink = 1.0 - (j / float(clip_grid))
            r_clip = shrink * rmax
            scale_clip = r_clip / qmax
            scale_clip_max = torch.amax(scale_clip.abs(), keepdim=True)
            scale_clip_scale = scale_clip_max / max_float
            scale_clip_unscaled = (scale_clip / scale_clip_scale)
            scale_clip_unscaled = torch.clamp(scale_clip_unscaled, min_float, max_float)
            clip_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(scale_clip_unscaled)) + bias_q)).detach(), 1.0)
            clip_scales = 2.0 ** (clip_log_scales - Mq - bias_q)
            scale_clip_unscaled = (scale_clip_unscaled / clip_scales).round()
            scale_clip_unscaled = scale_clip_unscaled * clip_scales
            scale_clip = scale_clip_unscaled * scale_clip_scale

            x_clip = w_base / scale_clip
            is_negative_scale_clip = sign_scale & (scale_clip < 0)
            idx_pos = torch.bucketize(x_clip, mids, right=False)
            idx_neg = torch.bucketize(x_clip, mids, right=True)
            idx = torch.where(is_negative_scale_clip, idx_neg, idx_pos)
            q_clip = vals[idx]
            err_clip = (q_clip - x_clip).to(torch.float32).pow(2).sum(-1)
            mask = err_clip < best_err
            best_err = torch.where(mask, err_clip, best_err)
            q_best = torch.where(mask.unsqueeze(-1), q_clip, q_best)
            scale_best = torch.where(mask.unsqueeze(-1), scale_clip, scale_best)
        w_q = q_best * scale_best

    if (group_size is None) or (group_size <= 0):
        return w_q
    else:
        return w_q.reshape(K, C)


@torch.no_grad()
def _quant_with_codebook_components(
    w_fp16,
    allow_value,
    group_size: Optional[int] = None,
    sign_scale: bool = False,
):
    """
    Quantize with a custom codebook and return
      - dequantized tensor (q * scale)
      - code-domain tensor q
      - per-group scale
    """
    allow_value = sorted(allow_value)

    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size()
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float16)

    vals = torch.tensor(allow_value, device=w_fp16_new.device, dtype=w_fp16_new.dtype)
    mids = (vals[:-1] + vals[1:]) * 0.5

    if sign_scale:
        vmin, vmax = torch.aminmax(w_fp16_new, dim=-1, keepdim=True)
        abs_vmax = vmax.abs()
        abs_vmin = vmin.abs()
        rmax = torch.where(abs_vmax >= abs_vmin, vmax, vmin)
    else:
        rmax = torch.amax(w_fp16_new.abs(), dim=-1, keepdim=True)

    qmax = max([abs(x) for x in allow_value])
    scale_fp = rmax / qmax

    scale_fp_max = torch.amax(scale_fp.abs(), keepdim=True)
    if torch.all(scale_fp_max == 0):
        q_tensor = torch.zeros_like(w_fp16_new)
        w_q = torch.zeros_like(w_fp16_new)
        if (group_size is None) or (group_size <= 0):
            return w_q, q_tensor, scale_fp
        return w_q.reshape(K, C), q_tensor.reshape(K, NUM_GROUP, group_size), scale_fp.reshape(K, NUM_GROUP, 1)

    # # Keep scale behavior consistent with quant_datatype / _quant_with_codebook
    # Mq = 6.0
    # Eq = 1.0
    # bias_q = 2 ** (Eq - 1) - 1
    # max_float = (2 - 2 ** (-Mq)) * 2 ** (2 ** Eq - 1 - bias_q)
    # min_float = -max_float
    # scale_fp_scale = scale_fp_max / max_float
    # scale_fp_unscaled = scale_fp / scale_fp_scale
    # scale_fp_unscaled = torch.clamp(scale_fp_unscaled, min_float, max_float)
    # tensor_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(scale_fp_unscaled)) + bias_q)).detach(), 1.0)
    # scales = 2.0 ** (tensor_log_scales - Mq - bias_q)
    # scale_fp_unscaled = (scale_fp_unscaled / scales).round()
    # scale_fp_unscaled = scale_fp_unscaled * scales
    # scale_fp = scale_fp_unscaled * scale_fp_scale

    safe_scale = torch.where(scale_fp == 0, torch.ones_like(scale_fp), scale_fp)
    x = w_fp16_new / safe_scale
    is_negative_scale = sign_scale & (scale_fp < 0)
    idx_pos = torch.bucketize(x, mids, right=False)
    idx_neg = torch.bucketize(x, mids, right=True)
    idx = torch.where(is_negative_scale, idx_neg, idx_pos)
    q_tensor = vals[idx]
    w_q = q_tensor * scale_fp

    if (group_size is None) or (group_size <= 0):
        return w_q, q_tensor, scale_fp
    return w_q.reshape(K, C), q_tensor.reshape(K, NUM_GROUP, group_size), scale_fp.reshape(K, NUM_GROUP, 1)

try:
    if hasattr(torch, "compile"):
        try:
            _quant_with_codebook = torch.compile(_quant_with_codebook, dynamic=True)
        except Exception:
            try:
                _quant_with_codebook = torch.compile(_quant_with_codebook)
            except Exception:
                pass
except Exception:
    pass


@torch.no_grad()
def search_datatype_axcore_pack(*args, **kwargs):
    raise ValueError(
        "Unsupported mixed datatype 'mixed_axcore'; AxCore kernels are not part of this Accuracy package"
    )


@torch.no_grad()
def search_datatype_unicore_auto_b32_greedy(w_fp16, wq_bits: int = 4, group_size: Optional[int] = None,
                                         sign_scale: bool = False,
                                         a_candidates: Optional[list] = None,
                                         budget: int = 32,
                                         return_meta: bool = False,
                                         logging: bool = False,
                                         clip: bool = False,
                                         clip_grid: int = 100,
                                         clip_maxshrink: float = 0.8):
    """
    Greedy budgeted UniCore DSE (precompute + iterative global gain).
    - Stage-1: precompute per-candidate per-group errors (errs_cand).
    - Build pairs (family, a) with pos/neg.
    - Iteratively select the pair with maximal global error reduction Δ, updating current_err.
    - Stage-2: reassign within chosen subset (no coverage flipping to match current behavior).
    """
    assert budget > 0, "budget must be positive"
    if wq_bits == 4:
        families = ('e1m2', 'e2m1', 'e3m0', 'e1m2_i')
    elif wq_bits == 3:
        families = ('e1m1', 'e2m0', 'e1m1_i', 'e2m0_i')
    

    # Shape & grouping
    if (group_size is None) or (group_size <= 0):
        K, C = w_fp16.size()
        NUM_GROUP = 1
        group_size_eff = C
    else:
        K, C = w_fp16.size()
        NUM_GROUP = C // group_size
        group_size_eff = group_size

    device = w_fp16.device
    w_view = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size_eff)

    # Prepare candidates
    if a_candidates is None:
        a_candidates = _generate_a_candidates(E=4, M=2)

    candidate_specs = []
    base_abs_sets = {fam: sorted(set(abs(v) for v in _unicore_base_codebook(fam))) for fam in families}
    for fam in families:
        base_abs = base_abs_sets[fam]
        for side in ('pos', 'neg'):
            for a in a_candidates:
                a_abs = abs(float(a))
                if any(abs(a_abs - b) < 1e-7 for b in base_abs):
                    continue
                codebook = _build_unicore_codebook(fam, side, a_abs)
                candidate_specs.append((f"unicore({fam},{side},a={a_abs})", codebook))

    # Name parser (local)
    def _parse_candidate_name(name: str):
        try:
            body = name[len("unicore("):-1]
            parts = body.split(',')
            fam = parts[0]; side = parts[1]; a_str = parts[2]
            a_val = float(a_str.split('=')[1])
            return {"pairable": True, "family": fam, "side": side, "a": a_val}
        except Exception:
            return {"pairable": False}

    # Stage-1: precompute errs per candidate
    errs_list = []
    for i, (name, codebook) in enumerate(candidate_specs):
        w_tmp = _quant_with_codebook(w_view, codebook, group_size=None, sign_scale=sign_scale)
        quant_error = (w_tmp - w_view).pow(2).mean(-1)  # [K, NUM_GROUP]
        errs_list.append(quant_error)
        del w_tmp, quant_error
    errs_cand = torch.stack(errs_list, dim=0)  # [N_cand, K, NUM_GROUP]
    del errs_list

    # Build pair list
    parsed = [_parse_candidate_name(name) for name, _ in candidate_specs]
    pairs = []  # list of (key, pos_idx, neg_idx)
    for idx, p in enumerate(parsed):
        if not p.get("pairable", False):
            continue
        # use dict to accumulate
    pair_map = {}
    for idx, p in enumerate(parsed):
        if not p.get("pairable", False):
            continue
        key = (p["family"], p["a"])
        d = pair_map.get(key, {})
        d[p["side"]] = idx
        pair_map[key] = d
    for key, d in pair_map.items():
        if ("pos" in d) and ("neg" in d):
            pairs.append((key, d["pos"], d["neg"]))

    if len(pairs) == 0:
        # Fallback to count-based if no complete pairs are available
        raise RuntimeError("No complete pairs are available")

    pair_budget = min(budget // 2, len(pairs))
    pos_idx_tensor = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)
    neg_idx_tensor = torch.tensor([p[2] for p in pairs], device=device, dtype=torch.long)

    # Greedy loop: maintain current_err (float32 for stability)
    current_err = None  # [K, NUM_GROUP] float32
    selected = torch.zeros(len(pairs), dtype=torch.bool, device=device)
    chosen_order = []  # store indices into pairs

    # Helper to compute pair_min for a chunk of pair indices
    def compute_pair_min(idx_chunk: torch.Tensor):
        errs_pos = errs_cand[pos_idx_tensor[idx_chunk]].to(torch.float32)  # [B, K, G]
        errs_neg = errs_cand[neg_idx_tensor[idx_chunk]].to(torch.float32)
        return torch.minimum(errs_pos, errs_neg)  # [B, K, G]

    # First pick: choose minimal sum(err_pair_min) to avoid inf baseline issues.
    # Use adaptive chunking to bound temporary [B, K, G] allocations.
    chunk = 256
    if device.type == 'cuda':
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device=device)
            # Rough upper bound of in-flight float32 buffers per pair in this stage:
            # errs_pos, errs_neg, pair_min, delta and a small margin.
            bytes_per_pair = max(1, int(K) * int(NUM_GROUP) * 4 * 6)
            target_bytes = max(64 * 1024 * 1024, int(free_bytes * 0.35))
            chunk = max(1, min(chunk, target_bytes // bytes_per_pair))
        except Exception:
            pass
    if logging:
        print(f"[mixed_unicore_auto_b{budget}_greedy] pair-search chunk={chunk}")

    best_idx = -1
    best_sum = None
    for start in range(0, len(pairs), chunk):
        end = min(start + chunk, len(pairs))
        idx_chunk = torch.arange(start, end, device=device)
        pair_min = compute_pair_min(idx_chunk)  # [B, K, G]
        sums = pair_min.sum(dim=(1, 2))  # [B]
        if best_sum is None:
            best_sum, local_idx = torch.min(sums, dim=0)
            best_idx = start + int(local_idx.item())
        else:
            cur_min, local_idx = torch.min(sums, dim=0)
            if cur_min.item() < best_sum.item():
                best_sum = cur_min
                best_idx = start + int(local_idx.item())
        del pair_min, sums

    if best_idx >= 0:
        chosen_order.append(best_idx)
        selected[best_idx] = True
        # set current_err to the selected pair's min error
        pair_min_best = compute_pair_min(torch.tensor([best_idx], device=device)).squeeze(0)
        current_err = pair_min_best  # float32 [K, G]
        del pair_min_best

    # Next picks: maximize gain = sum(relu(current_err - pair_min))
    while len(chosen_order) < pair_budget:
        best_gain = None
        best_idx2 = -1
        for start in range(0, len(pairs), chunk):
            end = min(start + chunk, len(pairs))
            idx_chunk = torch.arange(start, end, device=device)
            # mask out selected
            mask = ~selected[idx_chunk]
            if not torch.any(mask):
                continue
            idx_eff = idx_chunk[mask]
            pair_min = compute_pair_min(idx_eff)  # [B, K, G]
            delta = (current_err.unsqueeze(0) - pair_min).clamp_min(0.0)
            gains = delta.sum(dim=(1, 2))  # [B]
            if gains.numel() > 0:
                cur_gain, local_idx = torch.max(gains, dim=0)
                if (best_gain is None) or (cur_gain.item() > best_gain.item()):
                    best_gain = cur_gain
                    best_idx2 = int(idx_eff[int(local_idx.item())].item())
            del pair_min, delta, gains
        if best_idx2 < 0:
            # no more gains, but still fill up to budget using smallest sums among remaining
            filler_idx = None
            filler_sum = None
            for start in range(0, len(pairs), chunk):
                end = min(start + chunk, len(pairs))
                idx_chunk = torch.arange(start, end, device=device)
                mask = ~selected[idx_chunk]
                if not torch.any(mask):
                    continue
                idx_eff = idx_chunk[mask]
                pair_min = compute_pair_min(idx_eff)
                sums = pair_min.sum(dim=(1, 2))
                cur_min, local_idx = torch.min(sums, dim=0)
                if (filler_sum is None) or (cur_min.item() < filler_sum.item()):
                    filler_sum = cur_min
                    filler_idx = int(idx_eff[int(local_idx.item())].item())
                del pair_min, sums
            if filler_idx is None:
                break
            pick = filler_idx
        else:
            pick = best_idx2

        chosen_order.append(pick)
        selected[pick] = True
        pair_min_best = compute_pair_min(torch.tensor([pick], device=device)).squeeze(0)
        current_err = torch.minimum(current_err, pair_min_best)
        del pair_min_best

    # Build chosen set of candidate indices
    chosen_idx = set()
    chosen_pairs = [pairs[i] for i in chosen_order[:pair_budget]]
    for key, pos_i, neg_i in chosen_pairs:
        chosen_idx.add(pos_i)
        chosen_idx.add(neg_i)

    chosen_specs = [(i, candidate_specs[i]) for i in sorted(chosen_idx)]
    names_list = [item[1][0] for item in chosen_specs]
    if logging:
        print(f"[mixed_unicore_auto_b{budget}_greedy] Chosen candidate indices: {sorted(chosen_idx)}\n Candidate names: {' '.join(names_list)}")

    # Stage-2: reassign only within chosen subset (no coverage flipping).
    # Stream candidates to avoid holding all q_tmps/errs in memory at once.
    del errs_cand

    assign_local = torch.zeros((K, NUM_GROUP), dtype=torch.long, device=device)
    best_err = None
    q_tensor = None

    for li, (_, (name, codebook)) in enumerate(chosen_specs):
        w_tmp = _quant_with_codebook(
            w_view,
            codebook,
            group_size=None,
            sign_scale=sign_scale,
            clip_search=clip,
            clip_grid=clip_grid,
            clip_maxshrink=clip_maxshrink,
        )
        quant_error = (w_tmp - w_view).pow(2).mean(-1)  # [K, NUM_GROUP]

        if best_err is None:
            # Tie behavior is identical to argmin over stacked errors: first candidate wins ties.
            best_err = quant_error.clone()
            q_tensor = w_tmp
        else:
            better = quant_error < best_err
            if better.any():
                assign_local[better] = li
                best_err[better] = quant_error[better]
                q_tensor[better] = w_tmp[better].to(q_tensor.dtype)
            del better

        del w_tmp, quant_error

    if q_tensor is None:
        raise RuntimeError("No candidates selected in Stage-2")
    del best_err

    # Report final counts within the budgeted set
    final_counts_local = torch.bincount(assign_local.view(-1), minlength=len(chosen_specs)).to(torch.int64)
    summary = {}
    for li, cnt in enumerate(final_counts_local.tolist()):
        if cnt > 0:
            name = chosen_specs[li][1][0]
            summary[name] = cnt
    if logging:
        print(f"[mixed_unicore_auto_b{budget}_greedy] Final selection counts: {summary}")

    if not return_meta:
        return q_tensor.reshape(K, C)
    else:
        idx_to_name = [name for _, (name, _) in chosen_specs]
        meta = {
            'idx_to_name': idx_to_name,
            'final_choice_idx': assign_local.clone().cpu(),
            'group_shape': (int(K), int(NUM_GROUP)),
            'budget_indices': sorted(chosen_idx)
        }
        return q_tensor.reshape(K, C), meta

@torch.no_grad()
def quant_int_asym(w_fp16, wq_bits:int=4, group_size: Optional[int]=None):
    """
        Asymmetric INT quantization.
    """    
    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size() # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float16)
    
    rmin = torch.amin(w_fp16_new, dim=-1, keepdim=True)
    rmax = torch.amax(w_fp16_new, dim=-1, keepdim=True)
    qmin = 0
    qmax = 2**wq_bits - 1
    scale_fp = (rmax - rmin) / (qmax - qmin)
    scale_fp = scale_fp.clamp(min=1e-5, max=1e4)
    zeropoint = torch.round(-rmin / scale_fp).clamp(min=qmin, max=qmax)

    q_tensor = torch.clamp(torch.round(w_fp16_new / scale_fp) + zeropoint, min=qmin, max=qmax)

    w_fp16_new = (q_tensor - zeropoint) * scale_fp
    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)


@torch.no_grad()
def quant_mx(w_fp16, wq_bits:int=4, datatype: str="", group_size: int=32):
    """
        MX quantization.
        Reference: https://github.com/microsoft/microxcaling/blob/7bc41952de394f5cc5e782baf132e7c7542eb4e4/mx/mx_ops.py
    """ 
    if wq_bits == 3:
        DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT_MX
    elif wq_bits == 4:
        DATATYPE_MAPPING = DATATYPE_MAPPING_4_BIT_MX
    else:
        raise ValueError(f"Currently only support 3-bit, 4-bit quantization, not {wq_bits}-bit")

    assert datatype in DATATYPE_MAPPING, f"unexpected data type {datatype}."

    dtype = w_fp16.dtype
    allow_value = DATATYPE_MAPPING[datatype]
    mid_value = [(allow_value[i] + allow_value[i + 1]) / 2 for i in range(len(allow_value) - 1)]
    K, C = w_fp16.size() # output channel, input channel
    NUM_GROUP = C // group_size
    w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float32)
    
    shared_exp, _ = torch.max(w_fp16_new.abs(), dim=-1, keepdim=True)
    shared_exp = torch.floor(torch.log2(shared_exp))
    w_fp16_new = w_fp16_new / (2**shared_exp)
    qmax = max([abs(x) for x in allow_value])
    scale = 1 / (qmax / 2)
    x = w_fp16_new / scale

    q_tensor = torch.zeros_like(x)
    for i in range(len(allow_value)):
        data = allow_value[i]
        if i == 0:
            q_tensor += torch.where(x <= mid_value[i], data, 0)
        elif i == len(allow_value) - 1:
            q_tensor += torch.where(x > mid_value[i - 1], data, 0)
        else:
            q_tensor += torch.where((mid_value[i - 1] < x) & (x <= mid_value[i]), data, 0)

    w_fp16_new = q_tensor * scale * (2**shared_exp)
    return w_fp16_new.reshape(K, C).to(dtype)


@torch.no_grad()
def quant_nvfp(w_fp16, wq_bits:int=4, datatype: str="", group_size: int=16):
    if wq_bits == 3:
        DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT_NVFP
    elif wq_bits == 4:
        DATATYPE_MAPPING = DATATYPE_MAPPING_4_BIT_NVFP
    else:
        raise ValueError(f"Currently only support 3-bit, 4-bit quantization, not {wq_bits}-bit")

    assert datatype in DATATYPE_MAPPING, f"unexpected data type {datatype}."

    dtype = w_fp16.dtype
    allow_value = DATATYPE_MAPPING[datatype]
    mid_value = [(allow_value[i] + allow_value[i + 1]) / 2 for i in range(len(allow_value) - 1)]
    K, C = w_fp16.size() # output channel, input channel
    NUM_GROUP = C // group_size
    w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float32)
    
    tensor_rmax = torch.max(w_fp16_new.abs().to(torch.float32))
    tensor_scale_max = 448 # fp8_e4m3fn range
    tensor_scale = tensor_rmax / tensor_scale_max
    w_fp16_new = w_fp16_new / tensor_scale # nvfp4 first scale to fp8 with fp32
    w_fp16_new = w_fp16_new.to(torch.float8_e4m3fn)
    w_fp16_new = w_fp16_new.to(dtype)
    rmax = torch.amax(w_fp16_new.abs(), dim=-1, keepdim=True)
    qmax = max([abs(x) for x in allow_value])
    scale_fp = rmax / qmax
    scale_fp = scale_fp.to(torch.float8_e4m3fn)
    scale_fp = scale_fp.to(dtype)
    x = w_fp16_new / scale_fp
    
    q_tensor = torch.zeros_like(x)
    for i in range(len(allow_value)):
        data = allow_value[i]
        if i == 0:
            q_tensor += torch.where(x <= mid_value[i], data, 0)
        elif i == len(allow_value) - 1:
            q_tensor += torch.where(x > mid_value[i - 1], data, 0)
        else:
            q_tensor += torch.where((mid_value[i - 1] < x) & (x <= mid_value[i]), data, 0)

    w_fp16_new = q_tensor * scale_fp * tensor_scale
    return w_fp16_new.reshape(K, C).to(dtype)


@torch.no_grad()
def quant_datatype(w_fp16, wq_bits:int=4, datatype: str="", group_size: Optional[int]=None, sign_scale=False):
    if wq_bits == 3:
        DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT
    elif wq_bits == 4:
        DATATYPE_MAPPING = DATATYPE_MAPPING_4_BIT
    elif wq_bits == 5:
        DATATYPE_MAPPING = DATATYPE_MAPPING_5_BIT
    elif wq_bits == 6:
        DATATYPE_MAPPING = DATATYPE_MAPPING_6_BIT
    else:
        raise ValueError(f"Currently only support 3-, 4-, 5-, and 6-bit quantization, not {wq_bits}-bit")

    assert datatype in DATATYPE_MAPPING, f"unexpected data type {datatype}."

    dtype = w_fp16.dtype

    allow_value = DATATYPE_MAPPING[datatype]
    mid_value = [(allow_value[i] + allow_value[i + 1]) / 2 for i in range(len(allow_value) - 1)]

    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(dtype)
    else:
        K, C = w_fp16.size() # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(dtype)

    if sign_scale:
        vmin, vmax = torch.aminmax(w_fp16_new, dim=-1, keepdim=True)
        abs_vmax = vmax.abs()
        abs_vmin = vmin.abs()
        rmax = torch.where(abs_vmax >= abs_vmin, vmax, vmin)
    else:
        rmax = torch.amax(w_fp16_new.abs(), dim=-1, keepdim=True)
    qmax = max([abs(x) for x in allow_value])
    scale_fp = rmax / qmax
    # scale_fp = scale_fp.clamp(min=1e-5, max=1e4)
    
    scale_fp_max = torch.amax(scale_fp.abs(), keepdim=True)
    # fp scale
    M = 6.
    E = 1.
    bias = 2 ** (E - 1) - 1
    max_float = (2 - 2 ** (-M)) * 2 ** (2**E - 1 - bias)
    min_float = -max_float
    # calculate scaling factor
    scale_fp_scale = scale_fp_max / max_float
    # unscale tensor
    scale_fp_unscaled = (scale_fp / scale_fp_scale)
    # mapping to fp range
    scale_fp_unscaled = torch.clamp(scale_fp_unscaled, min_float, max_float)
    tensor_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(scale_fp_unscaled)) + bias)).detach(), 1.0)
    scales = 2.0 ** (tensor_log_scales - M - bias)
    scale_fp_unscaled = (scale_fp_unscaled / scales).round()
    scale_fp_unscaled = scale_fp_unscaled * scales
    # dequantization
    scale_fp = scale_fp_unscaled * scale_fp_scale
    
    x = w_fp16_new / scale_fp
    
    q_tensor = torch.zeros_like(x)
    # Check if scale is negative (when sign_scale is used and rmax is negative)
    is_negative_scale = sign_scale & (scale_fp < 0)
    # print(f"is_negative_scale.shape {is_negative_scale.shape}")
    for i in range(len(allow_value)):
        data = allow_value[i]
        if i == 0:
            # For positive: x <= mid_value[i], for negative scale: x < mid_value[i]
            condition = torch.where(is_negative_scale, x < mid_value[i], x <= mid_value[i])
            q_tensor += torch.where(condition, data, 0)
        elif i == len(allow_value) - 1:
            # For positive: x > mid_value[i-1], for negative scale: x >= mid_value[i-1]
            condition = torch.where(is_negative_scale, x >= mid_value[i - 1], x > mid_value[i - 1])
            q_tensor += torch.where(condition, data, 0)
        else:
            # For positive scale: mid_value[i-1] < x <= mid_value[i]
            # For negative scale: mid_value[i-1] <= x < mid_value[i]
            condition = torch.where(is_negative_scale,
                                   (mid_value[i - 1] <= x) & (x < mid_value[i]),
                                   (mid_value[i - 1] < x) & (x <= mid_value[i]))
            q_tensor += torch.where(condition, data, 0)
    
    mse = False
    maxshrink = 0.8
    grid = 100 
    if mse:
        best = torch.full([x.shape[0], x.shape[1]], float('inf'), device=w_fp16_new.device, dtype=w_fp16_new.dtype)
        for j in range(int(maxshrink * grid)):
            p = 1 - j / grid
            rmax1 = p * rmax
            
            scale_fp1 = rmax1 / qmax
            x1 = w_fp16_new / scale_fp1
            q_tensor1 = torch.zeros_like(x1)
            is_negative_scale = sign_scale & (scale_fp1 < 0)
            
            for i in range(len(allow_value)):
                data = allow_value[i]
                if i == 0:
                    # For positive: x <= mid_value[i], for negative scale: x < mid_value[i]
                    condition = torch.where(is_negative_scale, x < mid_value[i], x <= mid_value[i])
                    q_tensor1 += torch.where(condition, data, 0)
                elif i == len(allow_value) - 1:
                    # For positive: x > mid_value[i-1], for negative scale: x >= mid_value[i-1]
                    condition = torch.where(is_negative_scale, x >= mid_value[i - 1], x > mid_value[i - 1])
                    q_tensor1 += torch.where(condition, data, 0)
                else:
                    # For positive scale: mid_value[i-1] < x <= mid_value[i]
                    # For negative scale: mid_value[i-1] <= x < mid_value[i]
                    condition = torch.where(is_negative_scale,
                                        (mid_value[i - 1] <= x) & (x < mid_value[i]),
                                        (mid_value[i - 1] < x) & (x <= mid_value[i]))
                    q_tensor1 += torch.where(condition, data, 0)
            q_err = (q_tensor1 - x1).abs_().pow_(2) # following GPTQ and GPTAQ
            err = torch.sum(q_err, -1)
            tmp = err < best
            if torch.any(tmp):
                best[tmp] = err[tmp]
                scale_fp[tmp] = scale_fp1[tmp]
                q_tensor[tmp] = q_tensor1[tmp]

    w_fp16_new = q_tensor * scale_fp 

    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)


@torch.no_grad()
def quant_datatype_double_scale(w_fp16, wq_bits:int=4, datatype: str="", group_size: Optional[int]=None, sign_scale=False):
    if wq_bits == 3:
        DATATYPE_MAPPING = DATATYPE_MAPPING_3_BIT
    elif wq_bits == 4:
        DATATYPE_MAPPING = DATATYPE_MAPPING_4_BIT
    elif wq_bits == 5:
        DATATYPE_MAPPING = DATATYPE_MAPPING_5_BIT
    elif wq_bits == 6:
        DATATYPE_MAPPING = DATATYPE_MAPPING_6_BIT
    else:
        raise ValueError(f"Currently only support 3-, 4-, 5-, and 6-bit quantization, not {wq_bits}-bit")

    assert datatype in DATATYPE_MAPPING, f"unexpected data type {datatype}."

    allow_value = DATATYPE_MAPPING[datatype]
    mid_value = [(allow_value[i] + allow_value[i + 1]) / 2 for i in range(len(allow_value) - 1)]

    if (group_size is None) or (group_size <= 0):
        w_fp16_new = w_fp16.to(torch.float16)
    else:
        K, C = w_fp16.size() # output channel, input channel
        NUM_GROUP = C // group_size
        w_fp16_new = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size).to(torch.float16)

    K, NUM_GROUP, group_size = w_fp16.size() # output channel, input channel
    from quant_utils.quant_utils import fake_quantize_quarter_E4M3
    tensor_rmax = torch.amax(w_fp16_new.abs().to(torch.float32).reshape(K, 1, NUM_GROUP * group_size), dim=-1, keepdim=True)
    group_size = None
    tensor_scale_max = 480.
    tensor_scale = tensor_rmax / tensor_scale_max
    w_fp16_new = w_fp16_new / tensor_scale
    w_fp16_new = fake_quantize_quarter_E4M3(w_fp16_new.to(torch.float16))
    # w_fp16_new = w_fp16_new.to(torch.float8_e4m3fn)
    # w_fp16_new = w_fp16_new.to(torch.float16)
    if sign_scale:
        vmin, vmax = torch.aminmax(w_fp16_new, dim=-1, keepdim=True)
        abs_vmax = vmax.abs()
        abs_vmin = vmin.abs()
        rmax = torch.where(abs_vmax >= abs_vmin, vmax, vmin)
    else:
        rmax = torch.amax(w_fp16_new.abs(), dim=-1, keepdim=True)
    qmax = max([abs(x) for x in allow_value])
    scale_fp = rmax / qmax
    scale_fp = fake_quantize_quarter_E4M3(scale_fp)
    # scale_fp = scale_fp.to(torch.float8_e4m3fn)
    # scale_fp = scale_fp.to(torch.float16)
    x = w_fp16_new / scale_fp

    q_tensor = torch.zeros_like(x)
    # Check if scale is negative (when sign_scale is used and rmax is negative)
    is_negative_scale = sign_scale & (scale_fp < 0)
    # print(f"is_negative_scale.shape {is_negative_scale.shape}")
    for i in range(len(allow_value)):
        data = allow_value[i]
        if i == 0:
            # For positive: x <= mid_value[i], for negative scale: x < mid_value[i]
            condition = torch.where(is_negative_scale, x < mid_value[i], x <= mid_value[i])
            q_tensor += torch.where(condition, data, 0)
        elif i == len(allow_value) - 1:
            # For positive: x > mid_value[i-1], for negative scale: x >= mid_value[i-1]
            condition = torch.where(is_negative_scale, x >= mid_value[i - 1], x > mid_value[i - 1])
            q_tensor += torch.where(condition, data, 0)
        else:
            # For positive scale: mid_value[i-1] < x <= mid_value[i]
            # For negative scale: mid_value[i-1] <= x < mid_value[i]
            condition = torch.where(is_negative_scale,
                                   (mid_value[i - 1] <= x) & (x < mid_value[i]),
                                   (mid_value[i - 1] < x) & (x <= mid_value[i]))
            q_tensor += torch.where(condition, data, 0)

    w_fp16_new = q_tensor * scale_fp * tensor_scale
    w_fp16_new = w_fp16_new.to(torch.float16)

    if (group_size is None) or (group_size <= 0):
        return w_fp16_new
    else:
        return w_fp16_new.reshape(K, C)

@torch.no_grad()
def search_datatype(w_fp16, wq_bits:int=4, datatype: str='mixed_bitmod', group_size: Optional[int]=None):
    sign_scale = False
    # Fast path: dynamic UniCore without predefined 32 formats
    if ((wq_bits == 4) or (wq_bits == 3)) and (datatype.startswith(UNICORE_AUTO_PREFIX)):
        # parse budget and optional variant suffix, e.g.:
        #  - mixed_unicore_auto_b32         -> improvement-based (default)
        #  - mixed_unicore_auto_b32_clip    -> enable MSE clip search in Stage-2
        suffix = datatype.split(UNICORE_AUTO_PREFIX, 1)[1]
        # Extract leading digits as budget; default to 32 if missing
        budget = 32
        j = 0
        while j < len(suffix) and suffix[j].isdigit():
            j += 1
        if j > 0:
            try:
                budget = int(suffix[:j])
            except Exception:
                budget = 32
        rest = suffix[j:]
        clip = ('clip' in rest)
        return search_datatype_unicore_auto_b32_greedy(
            w_fp16,
            wq_bits=wq_bits,
            group_size=group_size,
            sign_scale=sign_scale,
            budget=budget,
            return_meta=False,
            clip=clip,
        )
    if wq_bits == 3:
        if datatype == 'mixed_bitmod':
            datatype_list = ['fp3_er_pos', 'fp3_er_neg', 'fp3_ea_pos', 'fp3_ea_neg']
        elif datatype == 'mixed_er':
            datatype_list = ['fp3_er_pos', 'fp3_er_neg']
        elif datatype == 'mixed_ea':
            datatype_list = ['fp3_ea_pos', 'fp3_ea_neg']
        elif datatype == 'mixed_ant':
            datatype_list = ['int3', 'fp3']
        elif datatype == UNICORE_DATATYPE:
            datatype_list = ['fp3_e1m1_4', 'fp3_e1m1_5', 'fp3_e1m1_6',
                             'fp3_e2m0_er_6', 'fp3_e2m0_10', 'fp3_e2m0_12', 'fp3_e2m0_14',
                             'fp3_e2m0_nz']
            # sign_scale = True
        else:
            raise ValueError(f"Unsupported 3-bit mixed datatype '{datatype}'")
    elif wq_bits == 4:
        if datatype == 'mixed_bitmod':
            datatype_list = [
                             'fp4_er_pos', 
                             'fp4_er_neg', 
                             'fp4_ea_pos', 
                             'fp4_ea_neg'
                             ]
        elif datatype == 'mixed_er':
            datatype_list = ['fp4_er_pos', 'fp4_er_neg']
        elif datatype == 'mixed_ea':
            datatype_list = ['fp4_ea_pos', 'fp4_ea_neg']
        elif datatype == 'mixed_ant':
            datatype_list = ['int4', 'flint4']
        elif datatype == 'mixed_mant':
            datatype_list = ['mant4_a0', 'mant4_a5', 'mant4_a10', 'mant4_a17',
                             'mant4_a20', 'mant4_a30', 'mant4_a40', 'mant4_a50',
                             'mant4_a60', 'mant4_a70', 'mant4_a80', 'mant4_a90',
                             'mant4_a100', 'mant4_a110', 'mant4_a120', 'int4']
        elif datatype == UNICORE_DATATYPE:
            datatype_list = [
                             'fp4_e1m2_4_pos', 'fp4_e1m2_4_neg', 
                             'fp4_e1m2_5_pos', 'fp4_e1m2_5_neg', 
                             'fp4_e1m2_6_pos', 'fp4_e1m2_6_neg',
                             'fp4_e1m2_7_pos', 'fp4_e1m2_7_neg',
                             'fp4_e2m1_7_pos', 'fp4_e2m1_7_neg', 
                             'fp4_e2m1_8_pos', 'fp4_e2m1_8_neg', 
                             'fp4_e2m1_10_pos', 'fp4_e2m1_10_neg',
                             'fp4_e2m1_12_pos', 'fp4_e2m1_12_neg',
                             'fp4_e2m1_14_pos', 'fp4_e2m1_14_neg',
                             'fp4_e3m0_nz',
                            'fp4_e1m2_er_025_pos', 'fp4_e1m2_er_025_neg',
                            'fp4_e1m2_er_075_pos', 'fp4_e1m2_er_075_neg',
                            'fp4_e1m2_er_125_pos', 'fp4_e1m2_er_125_neg',
                            'fp4_e2m1_er_025_pos', 'fp4_e2m1_er_025_neg',
                            'fp4_e2m1_er_250_pos', 'fp4_e2m1_er_250_neg',
                            'fp4_e2m1_er_500_pos', 'fp4_e2m1_er_500_neg',
                             ]
        else:
            raise ValueError(f"Unsupported 4-bit mixed datatype '{datatype}'")
    else:
        raise ValueError(f"Currently only support 3-bit and 4-bit mixed quantization, not {wq_bits}-bit")

    K, C = w_fp16.size() # output channel, input channel
    if (group_size is None) or (group_size <= 0):
        group_size = C
    NUM_GROUP = C // group_size
    w_fp16 = w_fp16.unsqueeze(-1).reshape(K, NUM_GROUP, group_size)
    q_tensor = torch.zeros_like(w_fp16)
    
    error = torch.full([K, NUM_GROUP], 1e3, dtype=w_fp16.dtype, device=w_fp16.device)
    best_datatype_idx = torch.zeros_like(error, dtype=torch.int32)
    for i, datatype in enumerate(datatype_list):
        w_fp16_tmp = quant_datatype(w_fp16, wq_bits=wq_bits, datatype=datatype, group_size=None, sign_scale=sign_scale)
        # w_fp16_tmp = quant_datatype_double_scale(w_fp16, wq_bits=wq_bits, datatype=datatype, group_size=None, sign_scale=sign_scale)
        quant_error = (w_fp16_tmp - w_fp16).pow(2).mean(-1)
        update_mask = torch.lt(quant_error, error)
        error[update_mask] = quant_error[update_mask]
        q_tensor[update_mask] = w_fp16_tmp[update_mask]
        best_datatype_idx[update_mask] = i

        del w_fp16_tmp, quant_error, update_mask
        
    datatype_counters = {dt: 0 for dt in datatype_list}
    for i, datatype in enumerate(datatype_list):
        count = (best_datatype_idx == i).sum().item()
        datatype_counters[datatype] = count
    # print(f"Datatype selection counts: {datatype_counters}")
    
    return q_tensor.reshape(K, C)


def quant_fp8(tensor, group_size, return_scale=False):
    '''
    fp8 e4m3 quantization
    
    Args:
        tensor: weight tensor to quantize
        group_size: group size for per-group quantization
        return_scale: if True, return (quantized_tensor, scale) without dequant; 
                      if False, return dequantized tensor (original behavior)
    '''
    per_tensor = False
    org_tensor_shape = tensor.shape
    if group_size > 0:
        assert org_tensor_shape[-1] % group_size == 0
        tensor = tensor.reshape(-1, group_size)
    elif per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2

    M = 3.
    E = 8. - 1 - M
    bias = 2 ** (E - 1) - 1
    max_float = (2 - 2 ** (-M)) * 2 ** (
            2**E - 1 - bias
        )
    min_float = -max_float
    max_val = tensor.abs().amax(dim=1, keepdim=True)
    S = max_val / max_float
    tensor_unscaled = (tensor / S)

    tensor_unscaled = torch.clamp(tensor_unscaled, min_float, max_float)
    subnormal_threshold = 2 ** (1 - bias)
    
    # Option 1: Push subnormals to threshold (original implementation)
    # tensor_unscaled = torch.where(tensor_unscaled > 0, 
    #                  torch.clamp(tensor_unscaled, min=subnormal_threshold, max=max_float),
    #                  torch.where(tensor_unscaled < 0,
    #                              torch.clamp(tensor_unscaled, min=-max_float, max=-subnormal_threshold),
    #                              tensor_unscaled))
    
    # Option 2: Set subnormals to zero, keep normals unchanged
    tensor_unscaled = torch.where(
        torch.abs(tensor_unscaled) < subnormal_threshold,
        torch.zeros_like(tensor_unscaled),
        tensor_unscaled
    )
    tensor_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(tensor_unscaled)) + bias)).detach(), 1.0)
    scales = 2.0 ** (tensor_log_scales - M - bias)
    tensor_q = (tensor_unscaled / scales).round()
    tensor_q = tensor_q * scales
    
    if return_scale:
        # Return quantized value and scale separately (for fused computation)
        # tensor_q is in FP8 range, S is the per-channel/per-group scale
        # Final value would be: tensor_q * S
        tensor_q = tensor_q.reshape(org_tensor_shape)
        if group_size > 0:
            # Per-group: S shape [out_features, in_features // group_size, 1]
            S = S.reshape(org_tensor_shape[0], -1, 1)
        # else: Per-channel: S shape remains [out_features, 1]
        return tensor_q, S
    
    tensor = tensor_q * S
    assert torch.isnan(tensor).sum() == 0
    tensor = tensor.reshape(org_tensor_shape)
    return tensor


def quant_fp4(tensor, group_size, return_scale=False):
    '''
    fp8 e2m1 quantization
    
    Args:
        tensor: weight tensor to quantize
        group_size: group size for per-group quantization
        return_scale: if True, return (quantized_tensor, scale) without dequant; 
                      if False, return dequantized tensor (original behavior)
    '''
    per_tensor = False
    org_tensor_shape = tensor.shape
    if group_size > 0:
        assert org_tensor_shape[-1] % group_size == 0
        tensor = tensor.reshape(-1, group_size)
    elif per_tensor:
        tensor = tensor.reshape(1, -1)
    assert tensor.dim() == 2

    M = 1.
    E = 4. - 1 - M
    bias = 2 ** (E - 1) - 1
    max_float = (2 - 2 ** (-M)) * 2 ** (
            2**E - 1 - bias
        )
    min_float = -max_float
    max_val = tensor.abs().amax(dim=1, keepdim=True)
    S = max_val / max_float
    zero_scale_mask = S == 0
    # Keep all-zero groups representable instead of producing 0/0 -> NaN.
    safe_S = torch.where(zero_scale_mask, torch.ones_like(S), S)
    tensor_unscaled = tensor / safe_S
    tensor_unscaled = torch.where(zero_scale_mask, torch.zeros_like(tensor_unscaled), tensor_unscaled)

    tensor_unscaled = torch.clamp(tensor_unscaled, min_float, max_float)
    subnormal_threshold = 2 ** (1 - bias)
    
    tensor_log_scales = torch.clamp((torch.floor(torch.log2(torch.abs(tensor_unscaled)) + bias)).detach(), 1.0)
    scales = 2.0 ** (tensor_log_scales - M - bias)
    tensor_q = (tensor_unscaled / scales).round()
    tensor_q = tensor_q * scales
    
    if return_scale:
        # Return quantized value and scale separately (for fused computation)
        # tensor_q is in FP8 range, S is the per-channel/per-group scale
        # Final value would be: tensor_q * S
        tensor_q = tensor_q.reshape(org_tensor_shape)
        if group_size > 0:
            # Per-group: S shape [out_features, in_features // group_size, 1]
            S = S.reshape(org_tensor_shape[0], -1, 1)
        # else: Per-channel: S shape remains [out_features, 1]
        return tensor_q, S
    
    tensor = tensor_q * S
    assert torch.isnan(tensor).sum() == 0
    tensor = tensor.reshape(org_tensor_shape)
    return tensor


def _cleanup_cuda_cache_if_needed():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _quantize_linear_weight_with_optional_cpu_search(
    linear_module: torch.nn.Linear,
    wq_bits: int,
    wq_datatype: str,
    wq_groupsize: Optional[int],
    search_on_cpu: bool,
    layerwise_offload: bool,
):
    """
    Quantize a single Linear layer with optional layer-wise CPU search to reduce
    GPU peak memory while keeping quantization math unchanged.
    """
    weight = linear_module.weight.data
    target_device = weight.device
    target_dtype = weight.dtype

    run_on_cpu = bool(search_on_cpu and target_device.type == 'cuda')

    if run_on_cpu and layerwise_offload:
        # Move current layer weight off GPU before search so temporaries stay on CPU.
        linear_module.weight.data = weight.to('cpu')
        del weight
        _cleanup_cuda_cache_if_needed()
        search_input = linear_module.weight.data
    elif run_on_cpu:
        search_input = weight.to('cpu')
    else:
        search_input = weight

    quantized = search_datatype(
        search_input,
        wq_bits=wq_bits,
        datatype=wq_datatype,
        group_size=wq_groupsize,
    )

    quantized = quantized.to(device=target_device, dtype=target_dtype)
    linear_module.weight.data = quantized

    del quantized
    if run_on_cpu and not layerwise_offload:
        del search_input
    gc.collect()
    _cleanup_cuda_cache_if_needed()


def quant_model(
    model,
    wq_bits: Optional[int] = None,
    wq_datatype: Optional[str] = None,
    wq_groupsize: Optional[int] = None,
    search_on_cpu: bool = False,
    layerwise_offload: bool = True,
):
    if (wq_datatype is None) or (wq_datatype in ["fp16", "fp32"]):
        print("Not applying quantization")
        time.sleep(2)
    elif (wq_datatype.startswith("int")) and ("asym" in wq_datatype):
        print(f"Applying asymmetric INT quantization with bits: {wq_bits}, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and 'lm_head' not in n and 'output_layer' not in n:
                print(f'Quantizing layer: {n}')
                m.weight.data = quant_int_asym(m.weight.data, wq_bits=wq_bits, group_size=wq_groupsize)
    elif (wq_datatype.startswith("int")) and ("asym" not in wq_datatype):
        print(f"Applying symmetric INT quantization with bits: {wq_bits}, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and 'lm_head' not in n and 'output_layer' not in n:
                print(f'Quantizing layer: {n}')
                m.weight.data = quant_int(m.weight.data, wq_bits=wq_bits, group_size=wq_groupsize)
    elif ("mx" in wq_datatype):
        '''
            We use hard-coded group size 32 based on the Open Compute Standard
            https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf
        '''
        print(f"Applying MX quantization with bits: {wq_bits}, datatype: {wq_datatype}, group size: 32")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and 'lm_head' not in n and 'output_layer' not in n:
                print(f'Quantizing layer: {n}')
                m.weight.data = quant_mx(m.weight.data, wq_bits=wq_bits, datatype=wq_datatype, group_size=32)
    elif ("nv" in wq_datatype):
        print(f"Applying NV quantization with bits: {wq_bits}, datatype: {wq_datatype}, group size: 16")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and 'lm_head' not in n and 'output_layer' not in n:
                print(f'Quantizing layer: {n}')
                m.weight.data = quant_nvfp(m.weight.data, wq_bits=wq_bits, datatype=wq_datatype, group_size=16)
    elif ("mixed" in wq_datatype) :
        if wq_datatype == 'mixed_axcore':
            raise ValueError("Unsupported mixed datatype 'mixed_axcore'; AxCore kernels are not part of this Accuracy package")
        print(f"Applying mixed datatype quantization with bits: {wq_bits}, datatype: {wq_datatype}, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and 'lm_head' not in n and 'output_layer' not in n:
                print(f'Quantizing layer: {n}')
                _quantize_linear_weight_with_optional_cpu_search(
                    m,
                    wq_bits=wq_bits,
                    wq_datatype=wq_datatype,
                    wq_groupsize=wq_groupsize,
                    search_on_cpu=search_on_cpu,
                    layerwise_offload=layerwise_offload,
                )
                gc.collect()
                _cleanup_cuda_cache_if_needed()
    elif ("fp8" in wq_datatype):
        print(f"Applying FP8 quantization with bits: {wq_bits}, datatype: {wq_datatype}, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and 'lm_head' not in n and 'output_layer' not in n:
                print(f'Quantizing layer: {n}')
                m.weight.data = quant_fp8(m.weight.data, group_size=wq_groupsize)
    # elif ("fp4" in wq_datatype):
    #     print(f"Applying FP4 quantization with bits: {wq_bits}, datatype: {wq_datatype}, group size: {wq_groupsize}")
    #     time.sleep(2)
    #     for n, m in model.named_modules():
    #         if isinstance(m, torch.nn.Linear) and 'lm_head' not in n and 'output_layer' not in n:
    #             print(f'Quantizing layer: {n}')
    #             m.weight.data = quant_fp4(m.weight.data, group_size=wq_groupsize)
    elif ("fp" in wq_datatype):
        print(f"Applying floating-point datatype quantization with bits: {wq_bits}, group size: {wq_groupsize}")
        time.sleep(2)
        for n, m in model.named_modules():
            if isinstance(m, torch.nn.Linear) and 'lm_head' not in n and 'output_layer' not in n:
                print(f'Quantizing layer: {n}')
                m.weight.data = quant_datatype(m.weight.data, wq_bits=wq_bits, datatype=wq_datatype, group_size=wq_groupsize)
    else:
        raise ValueError(f"Unsupported datatype {wq_datatype}")
