cmake_minimum_required(VERSION 2.8)

include_directories(
  src
)

add_executable(
  cpu
  cpu.cpp
  src/capturability.cpp
  src/capturability.h
)
