#!/usr/bin/env python
import numpy as np
import cupy as cp
import math

FOOTSIZE = 0.1
OMEGA = math.sqrt(9.81/0.50)
VELOCITY_OF_FOOT = 1.0
MINIMUM_DELTA_T = 0.1
GRID = 100
COUNT = 0

L_STEP_MIN = 0.0
L_STEP_MAX = 0.25
L_CP_MIN = -0.2
L_CP_MAX = 0.2
L_STRIDE_MIN = 0
L_STRIDE_MAX = 0.5

L_STEP = np.linspace(L_STEP_MIN, L_STEP_MAX, GRID)
L_CP = np.linspace(L_CP_MIN, L_CP_MAX, GRID)
LL, RR = np.meshgrid(L_STEP, L_CP)
SET_O = np.c_[LL.ravel(), RR.ravel()]

SET_I = np.linspace(L_STRIDE_MIN, L_STRIDE_MAX, 50)
