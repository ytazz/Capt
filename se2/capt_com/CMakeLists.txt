cmake_minimum_required(VERSION 3.0)

find_package(Eigen3 REQUIRED)

if(WIN32)
	set(Boost_USE_STATIC_LIBS        ON)
	set(Boost_USE_MULTITHREADED      ON)
endif()
find_package(Boost REQUIRED COMPONENTS date_time regex thread system)

set_property(GLOBAL PROPERTY USE_FOLDERS ON) 
set(CONF_FILES
	${PROJECT_SOURCE_DIR}/bin/conf/com.xml
)
source_group("conf" FILES ${CONF_FILES})

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable( capt_com
	main.cpp
	${CONF_FILES}
)

set_target_properties(capt_plot PROPERTIES DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX})
set_target_properties(capt_plot PROPERTIES RUNTIME_OUTPUT_DIRECTORY_DEBUG   ${BIN_INSTALL_DIR})
set_target_properties(capt_plot PROPERTIES RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BIN_INSTALL_DIR})
set_target_properties(capt_plot PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY    ${BIN_DIR})

target_link_libraries(capt_com capt_lib ${Boost_LIBRARIES})
