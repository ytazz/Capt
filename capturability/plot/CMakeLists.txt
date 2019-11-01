set(PLOT_SOURCES
  ${PROJECT_SOURCE_DIR}/plot/src/gnuplot.cpp
  ${PROJECT_SOURCE_DIR}/plot/src/cr_plot.cpp
  ${PROJECT_SOURCE_DIR}/plot/src/step_plot.cpp
)

# add_executable(animation animation.cpp ${CAPT_SOURCES} ${HEADERS} ${PLOT_SOURCES} ${PLOT_HEADERS})
add_executable(converter converter.cpp ${CAPT_SOURCES})
add_executable(cr_graph cr_graph.cpp ${CAPT_SOURCES} ${PLOT_SOURCES})
add_executable(step_graph step_graph.cpp ${CAPT_SOURCES} ${PLOT_SOURCES})

# target_link_libraries(animation ${EXPAT_LIBRARIES})
target_link_libraries(converter ${EXPAT_LIBRARIES})
target_link_libraries(cr_graph ${EXPAT_LIBRARIES})
target_link_libraries(step_graph ${EXPAT_LIBRARIES})
