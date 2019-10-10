cmake_minimum_required(VERSION 3.0)
project(Capt)

set(BUILD_GPU     false CACHE BOOL "build programs for GPU")
set(BUILD_CPU     false CACHE BOOL "build programs for CPU")
set(BUILD_EXAMPLE false CACHE BOOL "build sample programs")
set(MAKE_LIBRARY  false CACHE BOOL "make Capt library")

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
set(CUDA_SEPARABLE_COMPILATION ON)
if(${MAKE_LIBRARY})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -shared -fPIC")
endif()

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

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/cuda/include)
include_directories(${EXPAT_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

set(HEADERS
  ${PROJECT_SOURCE_DIR}/include/capturability.h
  ${PROJECT_SOURCE_DIR}/include/foot_planner.h
  ${PROJECT_SOURCE_DIR}/include/friction_filter.h
  ${PROJECT_SOURCE_DIR}/include/grid.h
  ${PROJECT_SOURCE_DIR}/include/interpolation.h
  ${PROJECT_SOURCE_DIR}/include/input.h
  ${PROJECT_SOURCE_DIR}/include/kinematics.h
  ${PROJECT_SOURCE_DIR}/include/loader.h
  ${PROJECT_SOURCE_DIR}/include/model.h
  ${PROJECT_SOURCE_DIR}/include/monitor.h
  ${PROJECT_SOURCE_DIR}/include/param.h
  ${PROJECT_SOURCE_DIR}/include/pendulum.h
  ${PROJECT_SOURCE_DIR}/include/planning.h
  ${PROJECT_SOURCE_DIR}/include/polygon.h
  ${PROJECT_SOURCE_DIR}/include/state.h
  ${PROJECT_SOURCE_DIR}/include/swing_foot.h
  ${PROJECT_SOURCE_DIR}/include/trajectory.h
  ${PROJECT_SOURCE_DIR}/include/vector.h
)

set(CPU_HEADERS
  ${PROJECT_SOURCE_DIR}/include/analysis_cpu.h
)

set(PLOT_HEADERS
  ${PROJECT_SOURCE_DIR}/include/cr_plot.h
  ${PROJECT_SOURCE_DIR}/include/plot_trajectory.h
  ${PROJECT_SOURCE_DIR}/include/gnuplot.h
)

set(SOURCES
  ${PROJECT_SOURCE_DIR}/src/capturability.cpp
  ${PROJECT_SOURCE_DIR}/src/cycloid.cpp
  ${PROJECT_SOURCE_DIR}/src/foot_planner.cpp
  ${PROJECT_SOURCE_DIR}/src/friction_filter.cpp
  ${PROJECT_SOURCE_DIR}/src/interpolation.cpp
  ${PROJECT_SOURCE_DIR}/src/grid.cpp
  ${PROJECT_SOURCE_DIR}/src/kinematics.cpp
  ${PROJECT_SOURCE_DIR}/src/loader.cpp
  ${PROJECT_SOURCE_DIR}/src/model.cpp
  ${PROJECT_SOURCE_DIR}/src/monitor.cpp
  ${PROJECT_SOURCE_DIR}/src/param.cpp
  ${PROJECT_SOURCE_DIR}/src/pendulum.cpp
  ${PROJECT_SOURCE_DIR}/src/planning.cpp
  ${PROJECT_SOURCE_DIR}/src/polygon.cpp
  ${PROJECT_SOURCE_DIR}/src/swing_foot.cpp
  ${PROJECT_SOURCE_DIR}/src/trajectory.cpp
  ${PROJECT_SOURCE_DIR}/src/vector.cpp
)

set(CPU_SOURCES
  ${PROJECT_SOURCE_DIR}/src/analysis_cpu.cpp
)

set(GPU_SOURCES
  ${PROJECT_SOURCE_DIR}/cuda/src/cuda_analysis.cu
  ${PROJECT_SOURCE_DIR}/cuda/src/cuda_vector.cu
)

set(PLOT_SOURCES
  ${PROJECT_SOURCE_DIR}/src/cr_plot.cpp
  # ${PROJECT_SOURCE_DIR}/src/plot_trajectory.cpp
  ${PROJECT_SOURCE_DIR}/src/gnuplot.cpp
)

if(${BUILD_CPU})
  add_executable( main_cpu
    main.cpp
    ${SOURCES}
    ${HEADERS}
    ${CPU_SOURCES}
    ${CPU_HEADERS}
  )
  target_link_libraries(main_cpu ${EXPAT_LIBRARIES})
endif()

if(${BUILD_GPU})
  cuda_add_executable( test_gpu
    test.cu
    ${SOURCES}
    ${GPU_SOURCES}
  )
  target_link_libraries(test_gpu ${EXPAT_LIBRARIES})
endif()

if(${MAKE_LIBRARY})
  add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})
  target_link_libraries(${PROJECT_NAME} ${EXPAT_LIBRARIES})
endif()

if(${BUILD_EXAMPLE})
  add_subdirectory(example)
endif()

configure_file(data/nao.xml ${CMAKE_CURRENT_BINARY_DIR}/bin COPYONLY)
configure_file(data/analysis.xml ${CMAKE_CURRENT_BINARY_DIR}/bin COPYONLY)
configure_file(data/param_val_xy.xml ${CMAKE_CURRENT_BINARY_DIR}/bin COPYONLY)
configure_file(data/val.xml ${CMAKE_CURRENT_BINARY_DIR}/bin COPYONLY)