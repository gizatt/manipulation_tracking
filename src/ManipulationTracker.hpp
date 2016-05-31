#ifndef MANIPULATION_TRACKER_H
#define MANIPULATION_TRACKER_H

#include <stdexcept>
#include <iostream>

#include "drake/systems/plants/RigidBodyTree.h"
#include "ManipulationTrackerCost.hpp"
#include "yaml-cpp/yaml.h"
#include <lcm/lcm-cpp.hpp>

// forward def
class ManipulationTrackerCost;

std::shared_ptr<RigidBodyTree> setupRobotFromConfig(YAML::Node config, Eigen::VectorXd& x0_robot, std::string base_path, bool verbose = false);

class ManipulationTracker {
public:
  ManipulationTracker(std::shared_ptr<const RigidBodyTree> robot, Eigen::Matrix<double, Eigen::Dynamic, 1> x0_robot_, std::shared_ptr<lcm::LCM> lcm_, bool verbose_ = false);
  ~ManipulationTracker() {};

  // register a cost function with the solver
  void addCost(std::shared_ptr<ManipulationTrackerCost> newCost){
    registeredCosts.push_back(newCost);
  }

  void update();
  Eigen::Matrix<double, Eigen::Dynamic, 1> output() { return x_robot; }

  // helper to publish out to lcm
  void publish();

private:
  std::shared_ptr<const RigidBodyTree> robot;
  std::vector<std::string> robot_names;
  KinematicsCache<double> robot_kinematics_cache;
  Eigen::Matrix<double, Eigen::Dynamic, 1> x;

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> sigma;


  std::shared_ptr<lcm::LCM> lcm;
  std::vector<std::shared_ptr<ManipulationTrackerCost>> registeredCosts;
  bool verbose;
};

#endif