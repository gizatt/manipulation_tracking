#ifndef GELSIGHT_COST_H
#define GELSIGHT_COST_H

#include <stdexcept>
#include <iostream>
#include "ManipulationTrackerCost.hpp"
#include "drake/multibody/rigid_body_tree.h"
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
  GelsightCost(std::shared_ptr<RigidBodyTree<double> > robot_, std::shared_ptr<lcm::LCM> lcm_, YAML::Node config);
  ~GelsightCost() {};
  bool constructCost(ManipulationTracker * tracker, const Eigen::Matrix<double, Eigen::Dynamic, 1> x_old, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Q, Eigen::Matrix<double, Eigen::Dynamic, 1>& f, double& K);

  int get_trans_with_utime(std::string from_frame, std::string to_frame,
                               long long utime, Eigen::Isometry3d & mat);
  void updateGelsightImage(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> new_gelsight_image);
  void handleGelsightFrameMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::image_t* msg);

  int get_input_num_pixel_cols() const { return input_num_pixel_cols; }
  int get_input_num_pixel_rows() const { return input_num_pixel_rows; }

  struct SensorPlane
  {
      Eigen::Vector3d lower_left; // v=0, u=0
      Eigen::Vector3d lower_right; // v=0, u=cols
      Eigen::Vector3d upper_left; // v=rows, u=0
      Eigen::Vector3d upper_right; // v=rows, u=cols
      Eigen::Vector3d normal;
      double thickness;
  };
  SensorPlane get_sensor_plane() const { return sensor_plane; }

private:
  SensorPlane sensor_plane;
  int sensor_body_id = -1;
  double downsample_amount = 20.0;
  double contact_threshold = 0.1;
  int input_num_pixel_cols = 256;
  int input_num_pixel_rows = 192;
  int num_pixel_cols, num_pixel_rows;

  double gelsight_depth_var = INFINITY;
  double gelsight_freespace_var = INFINITY;
  double max_considered_corresp_distance = 0.05;
  double min_considered_penetration_distance = 0.001; // TODO(gizatt): why is this necessary? Seems to prevent
                                                      // "sticking" behavior when pulling gelsight away from an
                                                      // object.
  double timeout_time = 0.5;
  bool verbose = false;

  std::shared_ptr<lcm::LCM> lcm;
  std::shared_ptr<RigidBodyTree<double> > robot;
  int nq;

  bot_lcmgl_t* lcmgl_gelsight_ = NULL;
  bot_lcmgl_t* lcmgl_corresp_ = NULL;

  std::mutex gelsight_frame_mutex;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> latest_gelsight_image;

  double lastReceivedTime;
  double startTime;
};

#endif
