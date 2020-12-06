cmake_minimum_required(VERSION 3.0)

find_package(EXPAT REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Springhead)
find_package(Scenebuilder)

include_directories(${EXPAT_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

set(CAPT_HEADERS
  ${PROJECT_SOURCE_DIR}/include/base.h
  ${PROJECT_SOURCE_DIR}/include/capturability.h
  ${PROJECT_SOURCE_DIR}/include/grid.h
  ${PROJECT_SOURCE_DIR}/include/input.h
  ${PROJECT_SOURCE_DIR}/include/plot.h
  ${PROJECT_SOURCE_DIR}/include/state.h
  ${PROJECT_SOURCE_DIR}/include/swing.h
)

set(CAPT_SOURCES
  ${PROJECT_SOURCE_DIR}/src/base.cpp
  ${PROJECT_SOURCE_DIR}/src/capturability.cpp
  ${PROJECT_SOURCE_DIR}/src/grid.cpp
  ${PROJECT_SOURCE_DIR}/src/plot.cpp
  ${PROJECT_SOURCE_DIR}/src/swing.cpp
)

add_library( capt_lib
${CAPT_SOURCES}
${CAPT_HEADERS}
)

target_link_libraries(capt_lib ${EXPAT_LIBRARIES})
target_link_libraries(capt_lib Springhead::Springhead)
target_link_libraries(capt_lib Scenebuilder)

