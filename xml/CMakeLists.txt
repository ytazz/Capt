cmake_minimum_required(VERSION 2.8.12)

project(xml)

# set(A_DIR "" CACHE PATH "comment")

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")

find_package(EXPAT REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${EXPAT_INCLUDE_DIRS})
include_directories(include)

set(_include
  include/reader.h
  include/vector.h
)

set(_src
  src/reader.cpp
  src/vector.cpp
)

add_executable(main main.cpp ${_src} ${_include})

add_executable(sample sample.cpp)
target_link_libraries(sample ${EXPAT_LIBRARIES})
