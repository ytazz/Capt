cmake_minimum_required(VERSION 3.0)
project(Capturability)

add_executable(anim_traj anim_traj.cpp ${SOURCES} ${HEADERS} ${PLOT_SOURCES} ${PLOT_HEADERS})

target_link_libraries(anim_traj ${EXPAT_LIBRARIES})