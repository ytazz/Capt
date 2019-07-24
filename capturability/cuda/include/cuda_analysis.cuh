#ifndef __CUDA_ANALYSIS_CUH__
#define __CUDA_ANALYSIS_CUH__

#include "cuda_grid.cuh"
#include "cuda_input.cuh"
#include "cuda_physics.cuh"
#include "cuda_polygon.cuh"
#include "cuda_state.cuh"
#include "cuda_vector.cuh"
#include "grid.h"
#include "nvidia.cuh"
#include <chrono>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <vector>

void initNstep(CA::Grid grid, int *cnstep);

void copyState(CA::Grid grid, CudaState *cstate);
void copyInput(CA::Grid grid, CudaInput *cinput);
void copyGrid(CA::Grid grid, CudaGrid *cgrid);

__global__ void exeZeroStep(CudaState *state, CudaInput *input, int *nstep,
                            CudaGrid *grid);

#endif // __CUDA_ANALYSIS_CUH__