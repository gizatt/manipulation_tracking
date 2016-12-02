cmake_minimum_required(VERSION 2.6.0)

cmake_policy(SET CMP0045 OLD)

set(POD_NAME manipulation_tracking)
include(cmake/pods.cmake)

set(EIGEN3_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include/eigen3)
include_directories(${EIGEN3_INCLUDE_DIR}) 
message("Eigen3 include dir" ${EIGEN3_INCLUDE_DIR})

add_subdirectory(src)
