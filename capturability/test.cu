#include "analysis_gpu.cuh"

const int BPG = 1024; // Blocks  Per Grid  (max: 65535)
const int TPB = 1024; // Threads Per Block (max: 1024)

int main(void) {
  exeNstep<<<BPG, TPB>>>();

  return 0;
}