# CMakeList.txt:

cmake_minimum_required (VERSION 2.8)

project ("Lab2")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
file(COPY data DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
add_executable( Lab2 Lab2.cpp )
target_link_libraries( Lab2 ${OpenCV_LIBS} )
