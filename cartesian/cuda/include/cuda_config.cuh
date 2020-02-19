#ifndef __CUDA_CONFIG_CUH__
#define __CUDA_CONFIG_CUH__

// Setting for Analysis
const bool enableDoubleSupport = false;

// Setting for Parallel Computing
const int BPG = 65535; // Blocks  Per Grid  (max: 65535)
const int TPB = 1024;  // Threads Per Block (max: 1024)

#endif