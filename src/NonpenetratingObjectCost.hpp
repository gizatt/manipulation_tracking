#ifndef NON_PENETRATING_OBJECT_COST_H
#define NON_PENETRATING_OBJECT_COST_H

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
#include "lcmtypes/vicon/body_t.hpp"
#include "lcmtypes/bot_core/image_t.hpp"
#include <kinect/kinect-utils.h>
#include <mutex>
#include <bot_lcmgl_client/lcmgl.h>
#include <bot_frames/bot_frames.h>
#include <bot_param/param_client.h>


class NonpenetratingObjectCost : public ManipulationTrackerCost {
public:
  NonpenetratingObjectCost(std::shared_ptr<RigidBodyTree> robot_, std::shared_ptr<lcm::LCM> lcm_, YAML::Node config);
  ~NonpenetratingObjectCost() {};
  bool constructCost(ManipulationTracker * tracker, const Eigen::Matrix<double, Eigen::Dynamic, 1> x_old, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Q, Eigen::Matrix<double, Eigen::Dynamic, 1>& f, double& K);

  void initBotConfig(const char* filename);
  int get_trans_with_utime(std::string from_frame, std::string to_frame,
                               long long utime, Eigen::Isometry3d & mat);
  //void handleSavePointcloudMsg(const lcm::ReceiveBuffer* rbuf,
  //                         const std::string& chan,
  //                         const bot_core::raw_t* msg);

private:
  double nonpenetration_var = INFINITY;
  bool verbose = false;
  bool verbose_lcmgl = false;
  int num_surface_pts = 500;
  double timeout_time = 0.5;

  std::shared_ptr<lcm::LCM> lcm;
  std::shared_ptr<RigidBodyTree> complete_robot_;

  std::shared_ptr<RigidBodyTree> robot;
  std::vector<int> collision_robot_indices;
  std::vector<int> collision_robot_state_indices;

  int object_index = 0;
  int object_state_index = 0;

  KinematicsCache<double> robot_kinematics_cache;
  int nq;

  //bot_lcmgl_t* lcmgl_lidar_ = NULL;
  //bot_lcmgl_t* lcmgl_icp_ = NULL;
  //bot_lcmgl_t* lcmgl_measurement_model_ = NULL;
  BotParam* botparam_ = NULL;
  BotFrames* botframes_ = NULL;

  //std::mutex latest_cloud_mutex;
  //std::mutex camera_offset_mutex;

  double lastReceivedTime;
};

#endif