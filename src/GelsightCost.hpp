#ifndef GELSIGHT_COST_H
#define GELSIGHT_COST_H

#include <stdexcept>
#include <iostream>
#include "ManipulationTrackerCost.hpp"
#include "drake/systems/plants/RigidBodyTree.h"
#include <lcm/lcm-cpp.hpp>
#include <memory>
#include <mutex>
#include "yaml-cpp/yaml.h"

#include "lcmtypes/bot_core/rigid_transform_t.hpp"
#include "lcmtypes/bot_core/raw_t.hpp"
#include "lcmtypes/kinect/frame_msg_t.hpp"
#include "lcmtypes/bot_core/image_t.hpp"
#include <kinect/kinect-utils.h>
#include <bot_lcmgl_client/lcmgl.h>


class GelsightCost : public ManipulationTrackerCost {
public:
  GelsightCost(std::shared_ptr<RigidBodyTree> robot_, std::shared_ptr<lcm::LCM> lcm_, YAML::Node config);
  ~GelsightCost() {};
  bool constructCost(ManipulationTracker * tracker, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Q, Eigen::Matrix<double, Eigen::Dynamic, 1>& f, double& K);

  int get_trans_with_utime(std::string from_frame, std::string to_frame,
                               long long utime, Eigen::Isometry3d & mat);
  void handleGelsightFrameMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::image_t* msg);


private:
  struct SensorPlane
  {
      Eigen::Vector3d lower_left; // v=0, u=0
      Eigen::Vector3d lower_right; // v=0, u=cols
      Eigen::Vector3d upper_left; // v=rows, u=0
      Eigen::Vector3d upper_right; // v=rows, u=cols
  };
  SensorPlane sensor_plane;
  int sensor_body_id = -1;
  double downsample_amount = 40.0;
  int input_num_pixel_cols = 640;
  int input_num_pixel_rows = 480;
  int num_pixel_cols, num_pixel_rows;

  double gelsight_depth_var = INFINITY;
  double gelsight_freespace_var = INFINITY;
  double timeout_time = 0.5;
  bool verbose = false;

  std::shared_ptr<lcm::LCM> lcm;
  std::shared_ptr<RigidBodyTree> robot;
  KinematicsCache<double> robot_kinematics_cache;
  int nq;

  bot_lcmgl_t* lcmgl_gelsight_ = NULL;

  std::mutex gelsight_frame_mutex;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> latest_gelsight_image;

  double lastReceivedTime;
};

#endif