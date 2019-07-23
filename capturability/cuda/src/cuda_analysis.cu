#include "cuda_analysis.cuh"

// __global__ void exeZero(Capturability *capturability, Grid *grid) {
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;
//   const unsigned int num_data = grid->getNumState() * grid->getNumInput();
//
//   while (tid < num_data) {
//     int state_id = tid / grid->getNumInput();
//     int input_id = tid % grid->getNumInput();
//
//     State state = grid->getState(state_id);
//     Input input = grid->getInput(input_id);
//
//     if (capturability->capturable(state, 0)) {
//       capturability->setCaptureSet(state_id, input_id, 0, 0);
//     }
//
//     tid += blockDim.x * gridDim.x;
//   }
// }

__global__ void exeNstep() {
  printf("a\n");
  vec2_t vec;
}