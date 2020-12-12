cmake_minimum_required(VERSION 3.0)

find_package(Eigen3 REQUIRED)

set_property(GLOBAL PROPERTY USE_FOLDERS ON) 
set(CONF_FILES
	${PROJECT_SOURCE_DIR}/bin/conf/sim.xml
)
source_group("conf" FILES ${CONF_FILES})

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable( capt_sim
	main.cpp
	${CONF_FILES}
)

set_target_properties(capt_sim PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_target_properties(capt_sim PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${BIN_INSTALL_DIR})
set_target_properties(capt_sim PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BIN_INSTALL_DIR})
set_target_properties(capt_sim PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY    ${BIN_DIR})

target_link_libraries(capt_sim capt_lib)
