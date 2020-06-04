cmake_minimum_required(VERSION 3.0)
project(Capt)

if(NOT CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX d)
endif()

set(BIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/bin)
if(UNIX)
	set(BIN_INSTALL_DIR ${BIN_DIR})
else()
	if(${CMAKE_GENERATOR_PLATFORM} STREQUAL "x64")
		set(BIN_INSTALL_DIR ${BIN_DIR}/x64)
	else()
		set(BIN_INSTALL_DIR ${BIN_DIR}/x86)
	endif()
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON) 
set(CONF_FILES
	${PROJECT_SOURCE_DIR}/data/valkyrie.xml
	${PROJECT_SOURCE_DIR}/data/valkyrie_xy.xml
)
source_group("conf" FILES ${CONF_FILES})


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

set(CAPT_HEADERS
  ${PROJECT_SOURCE_DIR}/include/base.h
  ${PROJECT_SOURCE_DIR}/include/capturability.h
  ${PROJECT_SOURCE_DIR}/include/grid.h
  ${PROJECT_SOURCE_DIR}/include/input.h
  ${PROJECT_SOURCE_DIR}/include/loader.h
  ${PROJECT_SOURCE_DIR}/include/model.h
#  ${PROJECT_SOURCE_DIR}/include/monitor.h
  ${PROJECT_SOURCE_DIR}/include/param.h
#  ${PROJECT_SOURCE_DIR}/include/pendulum.h
  ${PROJECT_SOURCE_DIR}/include/polygon.h
  ${PROJECT_SOURCE_DIR}/include/state.h
  ${PROJECT_SOURCE_DIR}/include/swing.h
#  ${PROJECT_SOURCE_DIR}/include/trajectory.h
)

set(CAPT_SOURCES
  ${PROJECT_SOURCE_DIR}/src/base.cpp
  ${PROJECT_SOURCE_DIR}/src/capturability.cpp
  ${PROJECT_SOURCE_DIR}/src/grid.cpp
  ${PROJECT_SOURCE_DIR}/src/loader.cpp
  ${PROJECT_SOURCE_DIR}/src/model.cpp
#  ${PROJECT_SOURCE_DIR}/src/monitor.cpp
  ${PROJECT_SOURCE_DIR}/src/param.cpp
#  ${PROJECT_SOURCE_DIR}/src/pendulum.cpp
  ${PROJECT_SOURCE_DIR}/src/polygon.cpp
  ${PROJECT_SOURCE_DIR}/src/swing.cpp
#  ${PROJECT_SOURCE_DIR}/src/trajectory.cpp
)

add_executable( main_cpu
main.cpp
${CAPT_SOURCES}
${CAPT_HEADERS}
${CONF_FILES}
)

set_target_properties(main_cpu PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_target_properties(main_cpu PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${BIN_INSTALL_DIR})
set_target_properties(main_cpu PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BIN_INSTALL_DIR})
set_target_properties(main_cpu PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY    ${BIN_DIR})

target_link_libraries(main_cpu ${EXPAT_LIBRARIES})

file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/dat)
