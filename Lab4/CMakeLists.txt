# CMakeList.txt:

cmake_minimum_required (VERSION 2.8)

project ("Lab4")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
file(COPY input.png DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_executable( Lab4 Lab4.cpp )
target_link_libraries( Lab4 ${OpenCV_LIBS} )
