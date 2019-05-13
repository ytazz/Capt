cmake_minimum_required(VERSION 3.0)
project(window)

set(QTDIR $ENV{QTDIR})
set(CMAKE_PREFIX_PATH ${QTDIR})
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")

find_package(Qt5Widgets REQUIRED)
find_package(EXPAT REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${EXPAT_INCLUDE_DIRS})

set(UI_SOURCES
  gui/mainwindow.ui
)

set(UI_RESOURCES
  resource/icons.qrc
)

# set(SOURCES
#   src/main.cpp
#   src/main_window.cpp
#   src/menu_widget.cpp
#   src/menu_button.cpp
#   src/setting_item.cpp
#   src/setting_widget.cpp
#   src/section.cpp
#   src/loader.cpp
#   src/model.cpp
#   src/vector.cpp
# )
file(GLOB_RECURSE SOURCES src/*.cpp)

# set(HEADERS
#   include/base.h
#   include/main_window.h
#   include/menu_widget.h
#   include/menu_button.h
#   include/setting_item.h
#   include/setting_widget.h
#   include/section.h
#   include/loader.h
#   include/model.h
#   include/vector.h
# )
file(GLOB_RECURSE HEADERS include/*.h)

qt5_wrap_ui(UI_HEADERS ${UI_SOURCES})
qt5_add_resources(RESOURCES ${UI_RESOURCES})

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} ${UI_HEADERS} ${RESOURCES})

target_link_libraries(${PROJECT_NAME} Qt5::Widgets ${EXPAT_LIBRARIES})

configure_file(data/nao.xml ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(data/graph.xml ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
