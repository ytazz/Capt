cmake_minimum_required(VERSION 3.0)
project(Filter)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")

add_executable(data data.cpp)
add_executable(lpf lpf.cpp)