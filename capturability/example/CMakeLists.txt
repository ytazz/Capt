cmake_minimum_required(VERSION 3.0)

add_executable(animation animation.cpp ${SOURCES} ${HEADERS} ${PLOT_SOURCES} ${PLOT_HEADERS})
add_executable(anim_cop anim_cop.cpp ${SOURCES} ${HEADERS} ${PLOT_SOURCES} ${PLOT_HEADERS})

target_link_libraries(animation ${EXPAT_LIBRARIES})
target_link_libraries(anim_cop ${EXPAT_LIBRARIES})