#ifndef __CUDA_ANALYSIS_CUH__
#define __CUDA_ANALYSIS_CUH__

#define NUM_STEP_MAX 5

#include "cuda_vector.cuh"
#include "cuda_memory_manager.cuh"
#include <vector>

namespace Cuda {

/* host function */

__host__ void saveBasin(std::string file_name, Condition cond, int *basin,
                        bool header = false);
__host__ void saveNStep(std::string file_name, Condition cond, int *nstep, int *trans,
                        bool header = false);
__host__ void saveCop(std::string file_name, Condition cond, Vector2 *cop,
                      bool header = false);
__host__ void saveStepTime(std::string file_name, Condition cond, double *step_time,
                           bool header = false);

/* device function */

__device__ bool    inPolygon(Vector2 point, Vector2 *convex, const int max_size, int swf_id);
__device__ bool    inPolygon(Vector2 point, Vector2 *vertex, int num_vertex);
__device__ Vector2 getClosestPoint(Vector2 point, Vector2 *vertex, int num_vertex);

__device__ State step(State state, Input input, Vector2 cop, double step_time, Physics *physics);
__device__ bool  existState(State state, GridCartesian *grid);
__device__ bool  existState(State state, GridPolar *grid);
__device__ int   getStateIndex(State state, GridCartesian *grid);
__device__ int   getStateIndex(State state, GridPolar *grid);
__device__ int   roundValue(double value);

/* global function */

__global__ void calcCop(State *state, GridCartesian *grid, Vector2 *foot_r, Vector2 *cop);
__global__ void calcStepTime(State *state, Input *input, GridCartesian *grid, double *step_time, Physics *physics);
__global__ void calcBasin(State *state, int *basin, GridCartesian *grid, Vector2 *foot_r, Vector2 *foot_l, Vector2 *convex);
__global__ void calcTrans(State *state, Input *input, int *trans, GridCartesian *grid,
                          Vector2 *cop, double *step_time, Physics *physics);
__global__ void calcTrans(State *state, Input *input, int *trans, GridPolar *grid,
                          Vector2 *cop, double *step_time, Physics *physics);


__global__ void exeNstep(int N, int *basin, int *nstep, int *trans, GridCartesian *grid);
__global__ void exeNstep(int N, int *basin, int *nstep, int *trans, GridPolar *grid);

} // namespace Cuda

#endif // __CUDA_ANALYSIS_CUH__