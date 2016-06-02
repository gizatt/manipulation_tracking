#ifndef DYNAMICS_COST_H
#define DYNAMICS_COST_H

#include <stdexcept>
#include <iostream>
#include "ManipulationTrackerCost.hpp"
#include "drake/systems/plants/RigidBodyTree.h"
#include <lcm/lcm-cpp.hpp>
#include <memory>
#include <mutex>
#include "yaml-cpp/yaml.h"

#include "lcmtypes/bot_core/robot_state_t.hpp"

class DynamicsCost : public ManipulationTrackerCost {
public:
  DynamicsCost(std::shared_ptr<const RigidBodyTree> robot_, std::shared_ptr<lcm::LCM> lcm_, YAML::Node config);
  ~DynamicsCost() {};
  bool constructPredictionMatrices(ManipulationTracker * tracker, const Eigen::Matrix<double, Eigen::Dynamic, 1> x_old, Eigen::Matrix<double, Eigen::Dynamic, 1>& x, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& P);

private:
  double dynamics_floating_base_var = INFINITY;
  double dynamics_other_var = INFINITY;
  bool verbose = false;

  std::shared_ptr<lcm::LCM> lcm;
  std::shared_ptr<const RigidBodyTree> robot;
  int nq;
};

#endif