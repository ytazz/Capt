# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kuribayashi/study/qt/window

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kuribayashi/study/qt/build-window-unknown-u65e2u5b9a

# Utility rule file for window_automoc.

# Include the progress variables for this target.
include CMakeFiles/window_automoc.dir/progress.make

CMakeFiles/window_automoc:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/kuribayashi/study/qt/build-window-unknown-u65e2u5b9a/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Automatic moc, uic and rcc for target window"
	/usr/bin/cmake -E cmake_autogen /home/kuribayashi/study/qt/build-window-unknown-u65e2u5b9a/CMakeFiles/window_automoc.dir/ ""

window_automoc: CMakeFiles/window_automoc
window_automoc: CMakeFiles/window_automoc.dir/build.make

.PHONY : window_automoc

# Rule to build all files generated by this target.
CMakeFiles/window_automoc.dir/build: window_automoc

.PHONY : CMakeFiles/window_automoc.dir/build

CMakeFiles/window_automoc.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/window_automoc.dir/cmake_clean.cmake
.PHONY : CMakeFiles/window_automoc.dir/clean

CMakeFiles/window_automoc.dir/depend:
	cd /home/kuribayashi/study/qt/build-window-unknown-u65e2u5b9a && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kuribayashi/study/qt/window /home/kuribayashi/study/qt/window /home/kuribayashi/study/qt/build-window-unknown-u65e2u5b9a /home/kuribayashi/study/qt/build-window-unknown-u65e2u5b9a /home/kuribayashi/study/qt/build-window-unknown-u65e2u5b9a/CMakeFiles/window_automoc.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/window_automoc.dir/depend

