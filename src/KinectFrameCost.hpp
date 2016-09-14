#ifndef KINECT_FRAME_COST_H
#define KINECT_FRAME_COST_H

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


class KinectFrameCost : public ManipulationTrackerCost {
public:
  KinectFrameCost(std::shared_ptr<RigidBodyTree> robot_, std::shared_ptr<lcm::LCM> lcm_, YAML::Node config);
  ~KinectFrameCost() {};
  bool constructCost(ManipulationTracker * tracker, const Eigen::Matrix<double, Eigen::Dynamic, 1> x_old, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Q, Eigen::Matrix<double, Eigen::Dynamic, 1>& f, double& K);

  void initBotConfig(const char* filename);
  int get_trans_with_utime(std::string from_frame, std::string to_frame,
                               long long utime, Eigen::Isometry3d & mat);
  void handleSavePointcloudMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::raw_t* msg);
  void handleKinectFrameMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const kinect::frame_msg_t* msg);
  void handleCameraOffsetMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const vicon::body_t* msg);

   // bounds to cut down point cloud, in world coords
  struct BoundingBox
  {
      double xMin = -100.;
      double xMax = 100.;
      double yMin = -100.;
      double yMax = 100.;
      double zMin = -100.;
      double zMax = 100.;
  };
  void setBounds(BoundingBox bounds) { pointcloud_bounds = bounds; }

private:
  double downsample_amount = 10.0;
  int input_num_pixel_cols = 640;
  int input_num_pixel_rows = 480;
  int num_pixel_cols, num_pixel_rows;

  double icp_var = INFINITY;
  double free_space_var = INFINITY;
  double max_considered_icp_distance = 0.075;
  double min_considered_joint_distance = 0.03;
  double timeout_time = 0.5;
  double max_scan_dist = 10.0;
  bool verbose = false;
  bool verbose_lcmgl = false;

  // We operate in one of four modes:
  // 1) Kinect has a body whose pose is learned. Have_camera_body is true. have_camera_body=true,world_frame=true
  // 2) Kinect listens to pose on LCM, have_camera_body=false but world_frame=true
  // 3) Kinect has hard coded world frame if world_frame=true. have_hardcoded_kinect2world = true
  // 3) Kinect frame is world frame. have_camera_body and world_frame = false
  bool have_camera_body_ = false;
  std::string camera_body_name_;
  int camera_body_ind_;
  bool world_frame = true;

  std::shared_ptr<lcm::LCM> lcm;
  std::shared_ptr<RigidBodyTree> robot;
  KinematicsCache<double> robot_kinematics_cache;
  int nq;

  bot_lcmgl_t* lcmgl_lidar_ = NULL;
  bot_lcmgl_t* lcmgl_icp_ = NULL;
  bot_lcmgl_t* lcmgl_measurement_model_ = NULL;
  BotParam* botparam_ = NULL;
  BotFrames* botframes_ = NULL;

  BoundingBox pointcloud_bounds;


  std::mutex latest_cloud_mutex;
  std::mutex camera_offset_mutex;
  Eigen::Isometry3d kinect2world_;
  bool have_hardcoded_kinect2world_ = false;
  Eigen::Isometry3d hardcoded_kinect2world_;

  KinectCalibration* kcal;
  Eigen::Matrix<double, 3, Eigen::Dynamic> latest_cloud;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> latest_depth_image;
  Eigen::Matrix<double, 3, Eigen::Dynamic> latest_color_image;
  Eigen::Matrix<double, 3, Eigen::Dynamic> raycast_endpoints;

  double lastReceivedTime;
  double last_got_kinect_frame;
};

#endif