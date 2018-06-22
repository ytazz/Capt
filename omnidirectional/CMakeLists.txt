cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

include_directories(
  src
)

cuda_add_executable(
  capturability_based_analysis
  src/root.cu
  src/capturability.cu
  src/capturability.h
)
