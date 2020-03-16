cmake_minimum_required(VERSION 2.8)

project (Rigid2D)

find_package (ITK REQUIRED)
include (${ITK_USE_FILE})

add_executable (Rigid2D Rigid2D.cxx)

target_link_libraries (Rigid2D ${ITK_LIBRARIES})
