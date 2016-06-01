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
  typedef std::pair<std::shared_ptr<ManipulationTrackerCost>, std::vector<int>> CostAndView;

  ManipulationTracker(std::shared_ptr<const RigidBodyTree> robot, Eigen::Matrix<double, Eigen::Dynamic, 1> x0_robot, std::shared_ptr<lcm::LCM> lcm, bool verbose = false);
  ~ManipulationTracker() {};

  // register a cost function with the solver
  void addCost(std::shared_ptr<ManipulationTrackerCost> new_cost);

  void update();
  Eigen::Matrix<double, Eigen::Dynamic, 1> output() { return x_; }

  // helper to publish out to lcm
  void publish();

private:
  std::shared_ptr<const RigidBodyTree> robot_;
  std::vector<std::string> robot_names_;
  KinematicsCache<double> robot_kinematics_cache_;


  Eigen::Matrix<double, Eigen::Dynamic, 1> x_;
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> covar_;


  std::shared_ptr<lcm::LCM> lcm_;

  // store all registered costs alongside a view into the variable
  // list, i.e. which decision vars that cost uses
  std::vector<CostAndView> registeredCostInfo_;

  bool verbose_;
};

#endif