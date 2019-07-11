#ifndef __ANALYSIS_CUH__
#define __ANALYSIS_CUH__

#include "capturability.h"
#include "grid.h"
#include "input.h"
#include "model.h"
#include "nvidia.cuh"
#include "param.h"
#include "pendulum.h"
#include "state.h"
#include "swing_foot.h"
#include "vector.h"
#include <chrono>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <vector>

using namespace CA;

__global__ void exeZero(Capturability *capturability, Grid *grid);

#endif // __ANALYSIS_CUH__