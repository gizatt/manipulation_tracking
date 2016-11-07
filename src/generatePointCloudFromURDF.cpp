#include <stdexcept>
#include <iostream>

#include "drake/systems/plants/RigidBodyTree.h"

#include <lcm/lcm-cpp.hpp>

#include "yaml-cpp/yaml.h"
#include "common/common.hpp"
#include "unistd.h"

#include <random>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>

#include "lcmtypes/bot_core/pointcloud_t.hpp"

using namespace std;
using namespace Eigen;

typedef pcl::PointXYZRGBA PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

static inline double randrange(double min, double max){
  return (((double)rand()) / RAND_MAX)*(max - min) + min;
}

int main(int argc, char** argv) {
  if (argc != 3){
    printf("Use: generatePointCloudFromURDF <path to URDF> <save_filename>\n");
    exit(-1);
  }

  std::shared_ptr<lcm::LCM> lcm(new lcm::LCM());
  if (!lcm->good()) {
    throw std::runtime_error("LCM is not good");
  }

  string urdfString = string(argv[1]);
  string outString = string(argv[2]);

  RigidBodyTree robot(urdfString);
  KinematicsCache<double> robot_kinematics_cache(robot.bodies);
  VectorXd q0_robot(robot.get_num_positions());
  q0_robot.setZero();
  robot_kinematics_cache.initialize(q0_robot);
  robot.doKinematics(robot_kinematics_cache);
  printf("Set up robot with %d positions\n", robot.get_num_positions());

  pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());

  const int kNumRays = 10000;
  Matrix3Xd origins(3, kNumRays);
  Matrix3Xd endpoints(3, kNumRays);
  VectorXd distances(kNumRays);
  for (int i=0; i < 10; i++){
    for (int k=0; k<kNumRays; k++){
      origins(0, k) = randrange(-100, 100);
      origins(1, k) = randrange(-100, 100);
      origins(2, k) = randrange(-100, 100);

      endpoints(0, k) = -origins(0, k);
      endpoints(1, k) = -origins(1, k);
      endpoints(2, k) = -origins(2, k);
    }
    robot.collisionRaycast(robot_kinematics_cache, origins, endpoints, distances, false);

  }

  lcm->handleTimeout(0);

  return 0;
}