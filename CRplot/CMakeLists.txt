cmake_minimum_required(VERSION 2.8)

project(CRplot)

set(NKK_DIR "" CACHE PATH "root directory of nkk")
include_directories(${NKK_DIR}/include)

set(NKK_LIB_DIR "" CACHE PATH "library directory of nkk")
link_directories(${NKK_LIB_DIR})

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_executable(make_animation
               src/Animation.cpp
               src/CRplot.cpp
               src/Gnuplot.cpp)

target_link_libraries(make_animation libnkk.so)
