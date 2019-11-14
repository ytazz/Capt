#ifndef __CUDA_ANALYSIS_CUH__
#define __CUDA_ANALYSIS_CUH__

#include "cuda_vector.cuh"
#include "cuda_memory_manager.cuh"
#include <vector>

namespace Cuda {

/* device function */

__device__ bool   inPolygon(vec2_t point, vec2_t *convex, const int max_size, int swf_id);
__device__ bool   inPolygon(vec2_t point, vec2_t *vertex, int num_vertex);
__device__ vec2_t getClosestPoint(vec2_t point, vec2_t *vertex, int num_vertex);

__device__ State step(State state, Input input, double step_time, Physics *physics);
__device__ bool  existState(State state, Grid *grid);
__device__ int   getStateIndex(State state, Grid *grid);
__device__ int   roundValue(double value);

/* global function */

__global__ void calcStepTime(State *state, Input *input, Grid *grid, double *step_time, Physics *physics);
__global__ void calcBasin(State *state, int *basin, Grid *grid, vec2_t *foot_r, vec2_t *foot_l, vec2_t *convex);
__global__ void calcTrans(State *state, Input *input, int *trans, Grid *grid,
                          double *step_time, Physics *physics);


__global__ void exeNstep(int N, int *basin, int *nstep, int *trans, Grid *grid);

} // namespace Cuda

#endif // __CUDA_ANALYSIS_CUH__