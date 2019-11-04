set(PLOT_SOURCES
  ${PROJECT_SOURCE_DIR}/plot/src/gnuplot.cpp
  ${PROJECT_SOURCE_DIR}/plot/src/cr_plot.cpp
  ${PROJECT_SOURCE_DIR}/plot/src/step_plot.cpp
)

add_executable(converter converter.cpp ${CAPT_SOURCES})
add_executable(cr_graph  cr_graph.cpp ${CAPT_SOURCES} ${PLOT_SOURCES})
add_executable(cr_gif    cr_gif.cpp ${CAPT_SOURCES} ${PLOT_SOURCES})
add_executable(step_graph step_graph.cpp ${CAPT_SOURCES} ${PLOT_SOURCES})

target_link_libraries(converter ${EXPAT_LIBRARIES})
target_link_libraries(cr_graph  ${EXPAT_LIBRARIES})
target_link_libraries(cr_gif    ${EXPAT_LIBRARIES})
target_link_libraries(step_graph ${EXPAT_LIBRARIES})