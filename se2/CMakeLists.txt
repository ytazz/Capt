cmake_minimum_required(VERSION 3.0)
project(Capt)

#set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3")

#set(CUDA_SEPARABLE_COMPILATION ON)
#set(CMAKE_INCLUDE_CURRENT_DIR ON)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

find_package(EXPAT REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EXPAT_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

set(CAPT_SOURCES
  ${PROJECT_SOURCE_DIR}/src/base.cpp
  ${PROJECT_SOURCE_DIR}/src/capturability.cpp
  ${PROJECT_SOURCE_DIR}/src/grid.cpp
  ${PROJECT_SOURCE_DIR}/src/model.cpp
  ${PROJECT_SOURCE_DIR}/src/monitor.cpp
  ${PROJECT_SOURCE_DIR}/src/param.cpp
  ${PROJECT_SOURCE_DIR}/src/pendulum.cpp
  ${PROJECT_SOURCE_DIR}/src/swing.cpp
  ${PROJECT_SOURCE_DIR}/src/trajectory.cpp
)

add_executable( main_cpu
main.cpp
${CAPT_SOURCES}
)
target_link_libraries(main_cpu ${EXPAT_LIBRARIES})

file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/dat)
