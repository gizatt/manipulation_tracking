#include <stdexcept>
#include <iostream>

#include "drake/multibody/rigid_body_tree.h"
#include "drake/multibody/parsers/urdf_parser.h"

#include <lcm/lcm-cpp.hpp>

#include "yaml-cpp/yaml.h"
#include "common/common.hpp"
#include "unistd.h"

#include <random>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "lcmtypes/bot_core/pointcloud_t.hpp"

using namespace std;
using namespace Eigen;
using namespace drake::parsers::urdf;

typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;

static inline double randrange(double min, double max){
  return (((double)rand()) / RAND_MAX)*(max - min) + min;
}

int main(int argc, char** argv) {
  srand(getUnixTime());

  if (argc != 3){
    printf("Use: generatePointCloudFromURDF <path to URDF> <save_filename_prefix>\n");
    exit(-1);
  }

  std::shared_ptr<lcm::LCM> lcm(new lcm::LCM());
  if (!lcm->good()) {
    throw std::runtime_error("LCM is not good");
  }

  string urdfString = string(argv[1]);
  string outStringPrefix = string(argv[2]);

  string outStringPts = outStringPrefix + string(".pts.pcd");
  string outStringNormals = outStringPrefix + string(".normals.pcd");

  RigidBodyTree<double> robot;
  AddModelInstanceFromUrdfFileWithRpyJointToWorld(urdfString, &robot);

  KinematicsCache<double> robot_kinematics_cache(robot.get_num_positions(), robot.get_num_velocities());
  VectorXd q0_robot(robot.get_num_positions());
  q0_robot.setZero();
  robot_kinematics_cache.initialize(q0_robot);
  robot.doKinematics(robot_kinematics_cache);
  printf("Set up robot with %d positions\n", robot.get_num_positions());

  pcl::PointCloud<PointType>::Ptr model_pts (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());

  const int kNumRays = 50000;
  
  Matrix3Xd origins(3, kNumRays);
  Matrix3Xd endpoints(3, kNumRays);
  VectorXd distances(kNumRays);
  Matrix3Xd normals(3, kNumRays);
  for (int k=0; k<kNumRays; k++){
    origins(0, k) = randrange(-100, 100);
    origins(1, k) = randrange(-100, 100);
    origins(2, k) = randrange(-100, 100);

    endpoints(0, k) = -origins(0, k);
    endpoints(1, k) = -origins(1, k);
    endpoints(2, k) = -origins(2, k);
  }
  vector<int> collision_body(kNumRays);
  robot.collisionRaycast(robot_kinematics_cache, origins, endpoints, distances, normals, collision_body, false);

  for (int k=0; k<kNumRays; k++){
    if (distances(k) >= 0 && collision_body[k] == 1){
      Vector3d dir = endpoints.block<3, 1>(0, k) - origins.block<3,1>(0, k);
      dir /= dir.norm();
      Vector3d pt = origins.block<3,1>(0, k) + dir * distances(k);
      model_pts->push_back(PointType( pt(0), pt(1), pt(2)));
      model_normals->push_back(NormalType( normals(0, k), normals(1, k), normals(2, k)));
    }
  }

  // publish current pointcloud to drake viewer
  bot_core::pointcloud_t msg;
  msg.utime = getUnixTime() * 1000 * 1000;
  msg.n_points = model_pts->size();
  msg.n_channels = 0;

  // unpack into float array
  vector<vector<float>> points(msg.n_points);
  for (size_t k=0; k<model_pts->size(); k++){
    points[k].push_back(model_pts->at(k).x);
    points[k].push_back(model_pts->at(k).y);
    points[k].push_back(model_pts->at(k).z);
  }
  msg.points = points;
  lcm->publish("DRAKE_POINTCLOUD_GENERATED_FROM_URDF", &msg);
  
  lcm->handleTimeout(0);

  pcl::visualization::PCLVisualizer viewer ("Point Collection");
  viewer.addPointCloud<PointType> (model_pts, "model pts and norms");
  //viewer.addPointCloudNormals<PointType, NormalType> (model_pts, model_normals, 1, 0.01, "model pts and norms");
  
  printf("Done, saving to %s and %s\n", outStringPts.c_str(), outStringNormals.c_str());
  pcl::io::savePCDFileASCII(outStringPts.c_str(), *model_pts);
  pcl::io::savePCDFileASCII(outStringNormals.c_str(), *model_normals);

  while (!viewer.wasStopped ())
    viewer.spinOnce ();

  return 0;
}
