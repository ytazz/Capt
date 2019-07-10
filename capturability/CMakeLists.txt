cmake_minimum_required(VERSION 3.0)
project(Capturability)

set(MAKE_LIBRARY  false CACHE BOOL "make capturability library")
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")
# if(${MAKE_LIBRARY})
#   set(CMAKE_CXX_FLAGS "-shared -fPIC")
# endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

find_package(EXPAT REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${EXPAT_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

set(HEADERS
  include/CA.h
  include/analysis.h
  include/capturability.h
  include/foot_planner.h
  include/friction_filter.h
  include/grid.h
  include/interpolation.h
  include/input.h
  include/kinematics.h
  include/loader.h
  include/model.h
  include/monitor.h
  include/param.h
  include/pendulum.h
  include/planning.h
  include/polygon.h
  include/state.h
  include/swing_foot.h
  include/trajectory.h
  include/vector.h
)
# file(GLOB_RECURSE HEADERS include/*.h)

set(PLOT_HEADERS
  include/cr_plot.h
  include/plot_trajectory.h
  include/gnuplot.h
)

set(SOURCES
  src/analysis.cpp
  src/capturability.cpp
  src/foot_planner.cpp
  src/friction_filter.cpp
  src/interpolation.cpp
  src/grid.cpp
  src/kinematics.cpp
  src/loader.cpp
  src/model.cpp
  src/monitor.cpp
  src/param.cpp
  src/pendulum.cpp
  src/planning.cpp
  src/polygon.cpp
  src/swing_foot.cpp
  src/trajectory.cpp
  src/vector.cpp
)
# file(GLOB_RECURSE SOURCES src/*.cpp)

set(PLOT_SOURCES
  src/cr_plot.cpp
  src/plot_trajectory.cpp
  src/gnuplot.cpp
)

add_executable(main main.cpp ${SOURCES} ${HEADERS})
add_executable(debug debug.cpp ${SOURCES} ${HEADERS})
add_executable(converter converter.cpp ${SOURCES} ${HEADERS})
add_executable(test_ik example/test_ik.cpp ${SOURCES} ${HEADERS})
add_executable(test_capture example/test_capture.cpp ${SOURCES} ${HEADERS})
add_executable(animation example/animation.cpp ${SOURCES} ${HEADERS} ${PLOT_SOURCES} ${PLOT_HEADERS})
# add_executable(anim_traj example/anim_traj.cpp ${SOURCES} ${HEADERS} ${PLOT_SOURCES} ${PLOT_HEADERS})
# add_executable(anim_friction example/anim_friction.cpp ${SOURCES} ${HEADERS} ${PLOT_SOURCES} ${PLOT_HEADERS})
add_executable(anim_push example/anim_push.cpp ${SOURCES} ${HEADERS} ${PLOT_SOURCES} ${PLOT_HEADERS})
add_executable(test_time example/test_time.cpp ${SOURCES} ${HEADERS})
add_executable(test_plan example/test_plan.cpp ${SOURCES} ${HEADERS})

target_link_libraries(main ${EXPAT_LIBRARIES})
target_link_libraries(debug ${EXPAT_LIBRARIES})
target_link_libraries(test_ik ${EXPAT_LIBRARIES})
target_link_libraries(test_capture ${EXPAT_LIBRARIES})
target_link_libraries(converter ${EXPAT_LIBRARIES})
target_link_libraries(animation ${EXPAT_LIBRARIES})
# target_link_libraries(anim_traj ${EXPAT_LIBRARIES})
# target_link_libraries(anim_friction ${EXPAT_LIBRARIES})
target_link_libraries(anim_push ${EXPAT_LIBRARIES})
target_link_libraries(test_time ${EXPAT_LIBRARIES})
target_link_libraries(test_plan ${EXPAT_LIBRARIES})

if(${MAKE_LIBRARY})
  add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})
  target_link_libraries(${PROJECT_NAME} ${EXPAT_LIBRARIES})
  # install(TARGETS ${PROJECT_NAME} DESTINATION ${PROJECT_SOURCE_DIR}/lib)
  # install(TARGETS ${PROJECT_NAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
endif()

configure_file(data/nao.xml ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
# configure_file(data/graph.xml ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(data/analysis.xml ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)