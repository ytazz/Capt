#ifndef __ANALYSIS_CUH__
#define __ANALYSIS_CUH__

// #include "capturability.cuh"
#include "grid.cuh"
#include "input.cuh"
#include "nvidia.cuh"
#include "polygon.cuh"
// #include "pendulum.cuh"
#include "state.cuh"
// #include "swing_foot.cuh"
#include "vector.cuh"
#include <chrono>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <vector>

// __global__ void exeZero(Capturability *capturability, Grid *grid);
// __global__ void func();
__device__ void deviceFunc();

__global__ void exeNstep();

#endif // __ANALYSIS_CUH__