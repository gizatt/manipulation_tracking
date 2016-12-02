find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenCV)

add_executable(minlp_detector minlp_detector.cpp)
pods_use_pkg_config_packages(minlp_detector lcm bot2-core bot2-lcmgl-client bot2-frames bot2-param-client kinect-utils drake-convex-hull eigen3 kinect-utils)
target_link_libraries(minlp_detector ObjectScan drakeRBM drakeOptimization drakeJoints drakeXMLUtil drakeCommon yaml-cpp ${PCL_LIBRARIES} z ${OpenCV_LIBS} pthread)
pods_install_executables(minlp_detector)
