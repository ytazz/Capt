cmake_minimum_required(VERSION 3.0)
project(Capt)

set(BUILD_GPU     false CACHE BOOL "build programs for GPU")
set(BUILD_CPU     false CACHE BOOL "build programs for CPU")
set(BUILD_EXAMPLE false CACHE BOOL "build sample programs")

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -O3")

set(CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)

find_package(EXPAT REQUIRED)
find_package(Eigen3 REQUIRED)
if(${BUILD_GPU})
  find_package(CUDA  QUIET REQUIRED)
  if(CUDA_FOUND)
    message("Found Cuda ! ( version: " ${CUDA_VERSION} " )")
  endif()
endif()

include_directories(${EXPAT_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/cuda/include)
include_directories(${PROJECT_SOURCE_DIR}/plot/include)

set(CAPT_SOURCES
  ${PROJECT_SOURCE_DIR}/src/base.cpp
  ${PROJECT_SOURCE_DIR}/src/capturability.cpp
  ${PROJECT_SOURCE_DIR}/src/config.cpp
  ${PROJECT_SOURCE_DIR}/src/cycloid.cpp
  ${PROJECT_SOURCE_DIR}/src/generator.cpp
  # ${PROJECT_SOURCE_DIR}/src/grid_map.cpp
  ${PROJECT_SOURCE_DIR}/src/grid.cpp
  ${PROJECT_SOURCE_DIR}/src/interpolation.cpp
  ${PROJECT_SOURCE_DIR}/src/loader.cpp
  ${PROJECT_SOURCE_DIR}/src/model.cpp
  ${PROJECT_SOURCE_DIR}/src/monitor.cpp
  ${PROJECT_SOURCE_DIR}/src/param.cpp
  ${PROJECT_SOURCE_DIR}/src/pendulum.cpp
  ${PROJECT_SOURCE_DIR}/src/planner.cpp
  ${PROJECT_SOURCE_DIR}/src/polygon.cpp
  ${PROJECT_SOURCE_DIR}/src/search.cpp
  ${PROJECT_SOURCE_DIR}/src/swing.cpp
  ${PROJECT_SOURCE_DIR}/src/timer.cpp
  ${PROJECT_SOURCE_DIR}/src/trajectory.cpp
  ${PROJECT_SOURCE_DIR}/src/tree.cpp
)

set(CPU_SOURCES
  ${PROJECT_SOURCE_DIR}/src/analysis_cpu.cpp
)

set(GPU_SOURCES
  ${PROJECT_SOURCE_DIR}/cuda/src/cuda_analysis.cu
  ${PROJECT_SOURCE_DIR}/cuda/src/cuda_memory_manager.cu
  ${PROJECT_SOURCE_DIR}/cuda/src/cuda_vector.cu
)

set(PLOT_SOURCES
  ${PROJECT_SOURCE_DIR}/plot/src/gnuplot.cpp
  ${PROJECT_SOURCE_DIR}/plot/src/cr_plot.cpp
  ${PROJECT_SOURCE_DIR}/plot/src/step_plot.cpp
  ${PROJECT_SOURCE_DIR}/plot/src/occupancy_plot.cpp
)

# add_executable(debug debug.cpp ${CAPT_SOURCES} ${PLOT_SOURCES})
# target_link_libraries(debug ${EXPAT_LIBRARIES})
# add_executable(check check.cpp ${CAPT_SOURCES} ${PLOT_SOURCES})
# target_link_libraries(check ${EXPAT_LIBRARIES})

if(${BUILD_CPU})
  add_executable( main_cpu
    main.cpp
    ${CAPT_SOURCES}
    ${PLOT_SOURCES}
    ${CPU_SOURCES}
  )
  target_link_libraries(main_cpu ${EXPAT_LIBRARIES})
endif()

if(${BUILD_GPU})
  cuda_add_executable( main_gpu
    main.cu
    ${CAPT_SOURCES}
    ${PLOT_SOURCES}
    ${GPU_SOURCES}
  )
  target_link_libraries(main_gpu ${EXPAT_LIBRARIES})
endif()

if(${BUILD_EXAMPLE})
  add_subdirectory(plot)
endif()

install(DIRECTORY data DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
install(DIRECTORY time DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/csv)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/cpu)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/gpu)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/dat)