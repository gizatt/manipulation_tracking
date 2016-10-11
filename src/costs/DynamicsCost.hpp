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
  DynamicsCost(std::shared_ptr<const RigidBodyTree> robot, std::shared_ptr<lcm::LCM> lcm, YAML::Node config);
  ~DynamicsCost() {};
  bool constructPredictionMatrices(ManipulationTracker * tracker, const Eigen::Matrix<double, Eigen::Dynamic, 1> x_old, Eigen::Matrix<double, Eigen::Dynamic, 1>& x, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& P);

private:
  struct DynamicsVars {
  	double floating_base_var = INFINITY;
  	double other_var = INFINITY;
  };

  DynamicsVars dynamics_vars_defaults_;

  std::map<std::string, DynamicsVars> dynamics_vars_per_robot_;
 
  bool verbose_ = false;

  std::shared_ptr<lcm::LCM> lcm_;
  std::shared_ptr<const RigidBodyTree> robot_;
  int nq_;
};

#endif