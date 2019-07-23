#include "cuda_analysis.cuh"

const int BPG = 1024; // Blocks  Per Grid  (max: 65535)
const int TPB = 1024; // Threads Per Block (max: 1024)

int main(void) {
  // Grid *grid, *dev_grid;
  // grid->num_state = 10;
  // HANDLE_ERROR(cudaMalloc((void **)&dev_grid, sizeof(Grid)));
  // HANDLE_ERROR(
  //     cudaMemcpy(dev_grid, grid, sizeof(Grid), cudaMemcpyHostToDevice));

  // exeNstep<<<BPG, TPB>>>();

  return 0;
}