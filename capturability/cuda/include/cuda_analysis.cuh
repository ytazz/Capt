#ifndef __CUDA_ANALYSIS_CUH__
#define __CUDA_ANALYSIS_CUH__

#include "cuda_grid.cuh"
#include "cuda_input.cuh"
#include "cuda_physics.cuh"
#include "cuda_polygon.cuh"
#include "cuda_state.cuh"
#include "cuda_vector.cuh"
#include "nvidia.cuh"
#include <chrono>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <vector>

__global__ void exeNstep();

#endif // __CUDA_ANALYSIS_CUH__