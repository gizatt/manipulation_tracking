set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

add_executable(test_2D_sdf test_2D_sdf.cpp)
target_link_libraries(test_2D_sdf)
pods_use_pkg_config_packages(test_2D_sdf eigen3)
pods_install_executables(test_2D_sdf)

add_executable(testGelsightCost testGelsightCost.cpp)
target_link_libraries(testGelsightCost stdc++ pthread GL GLU glut GLEW ManipulationTracker RobotStateCost GelsightCost JointStateCost KinectFrameCost AttachedApriltagCost DynamicsCost OptotrakMarkerCost NonpenetratingObjectCost GelsightOpenGLSim yaml-cpp bot2-core bot2-frames bot2-param-client drakeRBM drakeJoints drakeCollision drakeXMLUtil drakeCommon)
pods_install_executables(testGelsightCost)

add_executable(testGelsightSim gelsight_sim.cpp)
target_link_libraries(testGelsightSim stdc++ pthread GL GLU glut GLEW)
pods_use_pkg_config_packages(testGelsightSim drake-shapes)
pods_install_executables(testGelsightSim)