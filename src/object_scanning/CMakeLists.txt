find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenCV)

add_library(ObjectScan SHARED ObjectScan.cpp)
pods_use_pkg_config_packages(ObjectScan lcm bot2-core bot2-lcmgl-client bot2-frames bot2-param-client kinect-utils eigen3)
target_link_libraries(ObjectScan yaml-cpp ${PCL_LIBRARIES} ${OpenCV_LIBS} pthread z)
pods_install_libraries(ObjectScan)
pods_install_headers(ObjectScan.hpp DESTINATION perception_manipulation_tracker)

add_executable(runObjectScanner runObjectScanner.cpp)
pods_use_pkg_config_packages(runObjectScanner lcm bot2-core bot2-lcmgl-client bot2-frames bot2-param-client kinect-utils drake-convex-hull eigen3 kinect-utils)
target_link_libraries(runObjectScanner ObjectScan drakeRBM drakeJoints drakeXMLUtil drakeCommon yaml-cpp ${PCL_LIBRARIES} z ${OpenCV_LIBS} pthread)
pods_install_executables(runObjectScanner)
