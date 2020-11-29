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

add_subdirectory(capt_lib)
add_subdirectory(capt_gen)
add_subdirectory(capt_plot)

