cmake_minimum_required(VERSION 3.0)
project(capturability)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(EXPAT REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${EXPAT_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

set(HEADERS
  include/analysis.h
  include/capturability.h
  include/friction_filter.h
  include/grid.h
  include/input.h
  include/kinematics.h
  include/loader.h
  include/model.h
  include/param.h
  include/pendulum.h
  include/polygon.h
  include/state.h
  include/swing_foot.h
  include/trajectory.h
  include/vector.h
)
# file(GLOB_RECURSE HEADERS include/*.h)

set(SOURCES
  src/analysis.cpp
  src/capturability.cpp
  src/friction_filter.cpp
  src/grid.cpp
  src/kinematics.cpp
  src/loader.cpp
  src/model.cpp
  src/param.cpp
  src/pendulum.cpp
  src/polygon.cpp
  src/swing_foot.cpp
  src/trajectory.cpp
  src/vector.cpp
)
# file(GLOB_RECURSE SOURCES src/*.cpp)

add_executable(${PROJECT_NAME} debug.cpp ${SOURCES} ${HEADERS})
# add_executable(${PROJECT_NAME} main.cpp ${SOURCES} ${HEADERS})
# add_executable(converter converter.cpp ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} ${EXPAT_LIBRARIES})
# target_link_libraries(converter ${EXPAT_LIBRARIES})

configure_file(data/nao.xml ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
# configure_file(data/graph.xml ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(data/analysis.xml ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)