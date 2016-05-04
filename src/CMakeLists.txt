set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(test_2D_sdf test_2D_sdf.cpp)
target_link_libraries(test_2D_sdf)
pods_use_pkg_config_packages(test_2D_sdf eigen3)
pods_install_executables(test_2D_sdf)

add_library(ManipulationTracker SHARED ManipulationTracker.cpp)
pods_use_pkg_config_packages(ManipulationTracker lcmtypes_drc_lcmtypes lcm drake-convex-hull kinect-utils lcmtypes_kinect bot2-lcmgl-client bot2-core bot2-frames bot2-param-client opencv yaml-cpp eigen3)
target_link_libraries(ManipulationTracker drakeRBM lcmtypes_kinect)
pods_install_libraries(ManipulationTracker)
pods_install_headers(ManipulationTracker.hpp DESTINATION control)

add_executable(runManipulationTracker runManipulationTracker.cpp)
target_link_libraries(runManipulationTracker ManipulationTracker)
pods_install_executables(runManipulationTracker)