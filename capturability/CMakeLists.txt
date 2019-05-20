cmake_minimum_required(VERSION 3.0)
project(capturability)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(EXPAT REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${EXPAT_INCLUDE_DIRS})

set(SOURCES
  src/grid.cpp
  src/loader.cpp
  src/model.cpp
  src/param.cpp
  src/vector.cpp
  src/pendulum.cpp
  src/grid.cpp
)
# file(GLOB_RECURSE SOURCES src/*.cpp)
set(HEADERS
  include/grid.h
  include/input.h
  include/loader.h
  include/model.h
  include/param.h
  include/vector.h
  include/pendulum.h
  include/state.h
  include/grid.h
)
# file(GLOB_RECURSE HEADERS include/*.h)
add_executable(${PROJECT_NAME} main.cpp ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} ${EXPAT_LIBRARIES})

configure_file(data/nao.xml ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
# configure_file(data/graph.xml ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(data/analysis.xml ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)