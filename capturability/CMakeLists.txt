cmake_minimum_required(VERSION 3.0)
project(capturability)

find_package(GLUT)
find_package(OpenGL)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -lglut -lGL -lGLU")

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(EXPAT REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${EXPAT_INCLUDE_DIRS})

set(HEADERS
  include/analysis.h
  include/capturability.h
  include/grid.h
  include/input.h
  include/loader.h
  include/model.h
  include/param.h
  include/pendulum.h
  include/polygon.h
  include/state.h
  include/swing_foot.h
  include/vector.h
)
# file(GLOB_RECURSE HEADERS include/*.h)

set(SOURCES
  src/analysis.cpp
  src/capturability.cpp
  src/grid.cpp
  src/loader.cpp
  src/model.cpp
  src/param.cpp
  src/pendulum.cpp
  src/polygon.cpp
  src/swing_foot.cpp
  src/vector.cpp
)
# file(GLOB_RECURSE SOURCES src/*.cpp)

# add_executable(${PROJECT_NAME} debug.cpp ${SOURCES} ${HEADERS})
# add_executable(${PROJECT_NAME} main.cpp ${SOURCES} ${HEADERS})
add_executable(${PROJECT_NAME} test_cop.cpp ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} ${EXPAT_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${GLUT_LIBRARY} ${OPENGL_LIBRARY})

configure_file(data/nao.xml ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
# configure_file(data/graph.xml ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(data/analysis.xml ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)