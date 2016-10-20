set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories( ./ )

add_subdirectory(costs)

add_library(ManipulationTracker SHARED ManipulationTracker.cpp)
pods_use_pkg_config_packages(ManipulationTracker drake-convex-hull eigen3 lcmtypes_bot2-core)
target_link_libraries(ManipulationTracker drakeRBM yaml-cpp bot2-core bot2-frames bot2-param-client)
pods_install_libraries(ManipulationTracker)
pods_install_headers(ManipulationTracker.hpp DESTINATION perception_manipulation_tracker)

add_executable(runManipulationTracker runManipulationTracker.cpp)
target_link_libraries(runManipulationTracker ManipulationTracker RobotStateCost GelsightCost JointStateCost KinectFrameCost AttachedApriltagCost DynamicsCost OptotrakMarkerCost NonpenetratingObjectCost yaml-cpp)
pods_install_executables(runManipulationTracker)

add_executable(runErrorCalculator runErrorCalculator.cpp)
pods_use_pkg_config_packages(runErrorCalculator lcm bot2-core bot2-lcmgl-client bot2-frames bot2-param-client kinect-utils opencv drake-convex-hull eigen3)
target_link_libraries(runErrorCalculator drakeRBM drakeJoints drakeXMLUtil drakeCommon yaml-cpp)
pods_install_executables(runErrorCalculator)

add_library(GelsightOpenGLSim SHARED GelsightOpenGLSim.cpp)
target_link_libraries(GelsightOpenGLSim ManipulationTracker stdc++ pthread GL GLU glut GLEW)
pods_use_pkg_config_packages(GelsightOpenGLSim drake-shapes eigen3)
pods_install_libraries(GelsightOpenGLSim)
pods_install_headers(GelsightOpenGLSim.hpp DESTINATION perception_manipulation_tracker)

add_subdirectory(test)