cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

include_directories(
  src
  test
)

cuda_add_executable(
  # test_exe
  # test/test.cu
  first
  first.cu
  src/capturability.cu
  src/capturability.h
)
