#ifndef IRB140_ESTIMATOR_H
#define IRB140_ESTIMATOR_H

#include <stdexcept>
#include <iostream>
#include <lcm/lcm-cpp.hpp>

#include <sys/select.h>
#include "drake/systems/plants/RigidBodyTree.h"
#include "drake/systems/plants/BotVisualizer.h"
#include "drake/systems/plants/RigidBodySystem.h"
#include "lcmtypes/drc/utime_t.hpp"
#include "lcmtypes/drake/lcmt_robot_state.hpp"
#include "lcmtypes/bot_core/planar_lidar_t.hpp"
#include "lcmtypes/bot_core/rigid_transform_t.hpp"
#include "lcmtypes/bot_core/raw_t.hpp"
#include "lcmtypes/kinect/frame_msg_t.hpp"
#include "lcmtypes/bot_core/robot_state_t.hpp"
#include "lcmtypes/bot_core/joint_state_t.hpp"
#include <kinect/kinect-utils.h>
#include <mutex>
#include <bot_lcmgl_client/lcmgl.h>
#include <bot_frames/bot_frames.h>
#include <bot_param/param_client.h>

static double getUnixTime(void)
{
    struct timespec tv;

    if(clock_gettime(CLOCK_REALTIME, &tv) != 0) return 0;

    return (tv.tv_sec + (tv.tv_nsec / 1000000000.0));
}

class IRB140Estimator {
public:
  ~IRB140Estimator() {}

  IRB140Estimator(std::shared_ptr<RigidBodyTree> arm, std::shared_ptr<RigidBodyTree> manipuland, Eigen::Matrix<double, Eigen::Dynamic, 1> x0_arm, 
    Eigen::Matrix<double, Eigen::Dynamic, 1> x0_manipuland, const char* filename, const char* state_channelname,
    bool transcribe_published_floating_base,
    const char* hand_state_channelname);
  void run() {
    while(1){
      for (int i=0; i < 100; i++)
        this->lcm.handleTimeout(0);

      double dt = getUnixTime() - last_update_time;
      if (dt > timestep){
        last_update_time = getUnixTime();
        this->update(dt);
      }
    }
  }

  void update(double dt);
  void performCompleteICP(Eigen::Isometry3d& kinect2world, Eigen::MatrixXd& depth_image, Eigen::Matrix3Xd& points);

  void setupSubscriptions(const char* state_channelname,
    const char* hand_state_channelname);
  void initBotConfig(const char* filename);
  int get_trans_with_utime(std::string from_frame, std::string to_frame,
                               long long utime, Eigen::Isometry3d & mat);

  void handleSavePointcloudMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::raw_t* msg);

  void handlePlanarLidarMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::planar_lidar_t* msg);

  void handleSpindleFrameMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::rigid_transform_t* msg);

  void handleKinectFrameMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const kinect::frame_msg_t* msg);

  void handleRobotStateMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::robot_state_t* msg);

  void handleLeftHandStateMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::joint_state_t* msg);

private:
  std::shared_ptr<RigidBodyTree> arm;
  std::shared_ptr<RigidBodyTree> manipuland;
  KinematicsCache<double> manipuland_kinematics_cache;

  Eigen::Matrix<double, Eigen::Dynamic, 1> x_arm;
  Eigen::Matrix<double, Eigen::Dynamic, 1> x_manipuland;
  Eigen::Matrix<double, Eigen::Dynamic, 1> lambda_manipuland;

  std::mutex x_manipuland_measured_mutex;
  bool transcribe_published_floating_base;
  Eigen::Matrix<double, Eigen::Dynamic, 1> x_manipuland_measured;
  std::vector<bool> x_manipuland_measured_known;

  std::mutex latest_cloud_mutex;
  KinectCalibration* kcal;
  Eigen::Matrix<double, 3, Eigen::Dynamic> latest_cloud;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> latest_depth_image;
  Eigen::Matrix<double, 3, Eigen::Dynamic> raycast_endpoints;

  double downsample_amount = 15.0;
  int input_num_pixel_cols = 640;
  int input_num_pixel_rows = 480;
  int num_pixel_cols, num_pixel_rows;

  double last_update_time;
  double timestep = 0.02; // 50 hz target to not overload director

  lcm::LCM lcm;
  bot_lcmgl_t* lcmgl_lidar_ = NULL;
  bot_lcmgl_t* lcmgl_manipuland_ = NULL;
  bot_lcmgl_t* lcmgl_icp_ = NULL;
  bot_lcmgl_t* lcmgl_measurement_model_ = NULL;
  BotParam* botparam_ = NULL;
  BotFrames* botframes_ = NULL;

  std::shared_ptr<Drake::BotVisualizer<Drake::RigidBodySystem::StateVector>> visualizer;

  // box and table
  /*
  double manip_x_bounds[2] = {0.45, 0.75};
  double manip_y_bounds[2] = {-0.1, 0.2};
  double manip_z_bounds[2] = {0.7, 1.05};
  */

  // just hand
  /*
  double manip_x_bounds[2] = {0.45, 0.75};
  double manip_y_bounds[2] = {-0.1, 0.2};
  double manip_z_bounds[2] = {1.15, 1.35};
  */

  // hand and box and table
  /*
  double manip_x_bounds[2] = {0.45, 0.75};
  double manip_y_bounds[2] = {-0.1, 0.2};
  double manip_z_bounds[2] = {0.7, 1.35};
  */

  // arm and box and table
  double manip_x_bounds[2] = {0.3, 1.0};
  double manip_y_bounds[2] = {-0.4, 0.4};
  double manip_z_bounds[2] = {0.7, 1.5};


};

#endif