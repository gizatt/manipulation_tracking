set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package( OpenCV REQUIRED )

include_directories( ./ )

add_subdirectory(costs)
add_subdirectory(object_scanning)
add_subdirectory(detection)

add_library(ManipulationTracker SHARED ManipulationTracker.cpp)
pods_use_pkg_config_packages(ManipulationTracker drake-convex-hull drake-multibody-parsers eigen3 lcmtypes_bot2-core)
target_link_libraries(ManipulationTracker drakeRBM yaml-cpp bot2-core bot2-frames bot2-param-client)
pods_install_libraries(ManipulationTracker)
pods_install_headers(ManipulationTracker.hpp DESTINATION perception_manipulation_tracker)

add_executable(runManipulationTracker runManipulationTracker.cpp)
pods_use_pkg_config_packages(runManipulationTracker drake-multibody-parsers)
target_link_libraries(runManipulationTracker ManipulationTracker RobotStateCost GelsightCost JointStateCost KinectFrameCost AttachedApriltagCost DynamicsCost OptotrakMarkerCost NonpenetratingObjectCost yaml-cpp)
pods_install_executables(runManipulationTracker)

add_executable(runErrorCalculator runErrorCalculator.cpp)
pods_use_pkg_config_packages(runErrorCalculator lcm bot2-core bot2-lcmgl-client bot2-frames bot2-param-client kinect-utils drake-convex-hull drake-multibody-parsers eigen3)
target_link_libraries(runErrorCalculator drakeRBM drakeJoints drakeCommon yaml-cpp ${OpenCV_LIBS} pthread)
pods_install_executables(runErrorCalculator)

find_package(PCL 1.5 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


add_executable(runObjectDetector runObjectDetector.cpp)
pods_use_pkg_config_packages(runObjectDetector lcm bot2-core bot2-lcmgl-client bot2-frames bot2-param-client kinect-utils drake-convex-hull eigen3 kinect-utils)
target_link_libraries(runObjectDetector drakeRBM drakeJoints drakeCommon yaml-cpp ${PCL_LIBRARIES} ${OpenCV_LIBS} pthread z)
pods_install_executables(runObjectDetector)

add_executable(generatePointCloudFromURDF generatePointCloudFromURDF.cpp)
pods_use_pkg_config_packages(generatePointCloudFromURDF lcm bot2-core bot2-lcmgl-client bot2-frames bot2-param-client kinect-utils drake-convex-hull drake-multibody-parsers eigen3)
target_link_libraries(generatePointCloudFromURDF drakeRBM drakeJoints drakeCommon yaml-cpp ${PCL_LIBRARIES} ${OpenCV_LIBS} pthread)
pods_install_executables(generatePointCloudFromURDF)

add_library(GelsightOpenGLSim SHARED GelsightOpenGLSim.cpp)
target_link_libraries(GelsightOpenGLSim ManipulationTracker stdc++ pthread GL GLU glut GLEW ${OpenCV_LIBS})
pods_use_pkg_config_packages(GelsightOpenGLSim drake-shapes eigen3)
pods_install_libraries(GelsightOpenGLSim)
pods_install_headers(GelsightOpenGLSim.hpp DESTINATION perception_manipulation_tracker)

add_subdirectory(test)
