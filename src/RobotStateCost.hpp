#ifndef ROBOT_STATE_COST_H
#define ROBOT_STATE_COST_H

#include <stdexcept>
#include <iostream>
#include "ManipulationTrackerCost.hpp"
#include "drake/systems/plants/RigidBodyTree.h"
#include <lcm/lcm-cpp.hpp>
#include <memory>
#include <mutex>
#include "yaml-cpp/yaml.h"

#include "lcmtypes/bot_core/robot_state_t.hpp"

class RobotStateCost : public ManipulationTrackerCost {
public:
  RobotStateCost(std::shared_ptr<RigidBodyTree> robot_, std::shared_ptr<lcm::LCM> lcm_, YAML::Node config);
  ~RobotStateCost() {};

  bool constructCost(ManipulationTracker * tracker, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Q, Eigen::Matrix<double, Eigen::Dynamic, 1>& f, double& K);

  void handleRobotStateMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::robot_state_t* msg);
private:
  std::string state_channelname = "EST_ROBOT_STATE";
  double joint_known_encoder_var = INFINITY;
  double joint_known_floating_base_var = INFINITY;
  double timeout_time = 0.5;
  bool verbose = false;
  bool transcribe_floating_base_vars = false;

  std::shared_ptr<lcm::LCM> lcm;
  lcm::Subscription * state_sub;
  std::shared_ptr<RigidBodyTree> robot;
  int nq;

  Eigen::Matrix<double, Eigen::Dynamic, 1> x_robot_measured;
  std::vector<bool> x_robot_measured_known;
  std::mutex x_robot_measured_mutex;

  double lastReceivedTime;
};

#endif