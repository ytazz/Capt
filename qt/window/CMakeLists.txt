cmake_minimum_required(VERSION 3.0)
project(window)

set(QTDIR $ENV{QTDIR})
set(CMAKE_PREFIX_PATH ${QTDIR})
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")

find_package(Qt5Widgets REQUIRED)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

set(UI_SOURCES
  gui/mainwindow.ui
)

set(UI_RESOURCES
  res/icons.qrc
)

set(SOURCES
  src/main.cpp
  src/main_window.cpp
  src/menu_widget.cpp
  src/menu_button.cpp
  src/setting_item.cpp
  src/setting_widget.cpp
  src/section.cpp
)

set(HEADERS
  include/base.h
  include/main_window.h
  include/menu_widget.h
  include/menu_button.h
  include/setting_item.h
  include/setting_widget.h
  include/section.h
)

qt5_wrap_ui(UI_HEADERS ${UI_SOURCES})
qt5_add_resources(RESOURCES ${UI_RESOURCES})

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} ${UI_HEADERS} ${RESOURCES})

target_link_libraries(${PROJECT_NAME} Qt5::Widgets)
