#ifndef __CUDA_ANALYSIS_CUH__
#define __CUDA_ANALYSIS_CUH__

#define NUM_STEP_MAX 5

#include "cuda_vector.cuh"
#include "cuda_memory_manager.cuh"

namespace Cuda {

/* host function */

__host__ void outputBasin(std::string file_name, Condition cond, int *basin,
                          bool header = true);
__host__ void outputNStep(std::string file_name, Condition cond, int *nstep, int *trans,
                          bool header = true);

__host__ void exeZeroStep(Capt::Grid grid, Capt::Model model, int *basin);

/* device function */

__device__ State step(State state, Input input, Vector2 cop, Physics *physics);
__device__ bool  existState(State state, GridCartesian *grid);
__device__ bool  existState(State state, GridPolar *grid);
__device__ int   getStateIndex(State state, GridCartesian *grid);
__device__ int   getStateIndex(State state, GridPolar *grid);
__device__ int   roundValue(double value);

/* global function */

__global__ void calcStateTrans(State *state, Input *input, int *trans, GridCartesian *grid,
                               Vector2 *cop, Physics *physics);
__global__ void calcStateTrans(State *state, Input *input, int *trans, GridPolar *grid,
                               Vector2 *cop, Physics *physics);

__global__ void exeNStep(int N, int *basin, int *nstep, int *trans, GridCartesian *grid);
__global__ void exeNStep(int N, int *basin, int *nstep, int *trans, GridPolar *grid);

} // namespace Cuda

#endif // __CUDA_ANALYSIS_CUH__