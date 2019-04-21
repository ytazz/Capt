cmake_minimum_required(VERSION 2.8.12)

project(xml)

# set(A_DIR "" CACHE PATH "comment")

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")

find_package(EXPAT REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${EXPAT_INCLUDE_DIRS})

set(_src
  # src/xml.cpp
)

add_executable(main main.cpp ${_src})

target_link_libraries(main ${EXPAT_LIBRARIES})
