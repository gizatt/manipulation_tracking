cmake_minimum_required(VERSION 2.6.0)

cmake_policy(SET CMP0045 OLD)

set(POD_NAME manipulation_tracking)
include(cmake/pods.cmake)

add_subdirectory(src)
