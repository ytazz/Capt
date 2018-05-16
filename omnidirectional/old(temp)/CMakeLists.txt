cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

include_directories(
  src
)

cuda_add_executable(
  1stepCap_calc
  1stepCap_calc.cu
  src/capturability.cu
  src/capturability.h
)
