cmake_minimum_required(VERSION 3.0)

add_executable(animation animation.cpp ${SOURCES} ${HEADERS} ${PLOT_SOURCES} ${PLOT_HEADERS})
add_executable(converter converter.cpp ${SOURCES} ${HEADERS} ${PLOT_SOURCES} ${PLOT_HEADERS})
add_executable(debug debug.cpp ${SOURCES} ${HEADERS} ${PLOT_SOURCES} ${PLOT_HEADERS})

target_link_libraries(animation ${EXPAT_LIBRARIES})
target_link_libraries(converter ${EXPAT_LIBRARIES})
target_link_libraries(debug ${EXPAT_LIBRARIES})