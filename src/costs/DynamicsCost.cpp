#undef NDEBUG
#include <assert.h> 
#include <fstream>
#include "DynamicsCost.hpp"
#include "drake/util/convexHull.h"
#include <cmath>
#include "common/common.hpp"

using namespace std;
using namespace Eigen;

DynamicsCost::DynamicsCost(std::shared_ptr<const RigidBodyTree<double> > robot, std::shared_ptr<lcm::LCM> lcm, YAML::Node config) :
    robot_(robot),
    lcm_(lcm),
    nq_(robot->get_num_positions())
{
  if (config["dynamics_floating_base_var"])
    dynamics_vars_defaults_.floating_base_var = config["dynamics_floating_base_var"].as<double>();
  if (config["dynamics_other_var"])
    dynamics_vars_defaults_.other_var = config["dynamics_other_var"].as<double>();
  if (config["verbose"])
    verbose_ = config["verbose"].as<bool>();

  if (config["robot_specific_vars"]){
    for (auto iter = config["robot_specific_vars"].begin();
              iter != config["robot_specific_vars"].end();
              iter++){
      DynamicsVars robot_vars;
      robot_vars.floating_base_var = iter->second["dynamics_floating_base_var"].as<double>();
      robot_vars.other_var = iter->second["dynamics_other_var"].as<double>();
      dynamics_vars_per_robot_[iter->second["robot"].as<string>()] = robot_vars;
      printf("Loaded vars %f, %f for robot %s\n", robot_vars.floating_base_var, robot_vars.other_var, iter->first.as<string>().c_str());
    }
  }

}

/***********************************************
            KNOWN POSITION HINTS
*********************************************/
bool DynamicsCost::constructPredictionMatrices(ManipulationTracker * tracker, const Eigen::Matrix<double, Eigen::Dynamic, 1> x_old, Eigen::Matrix<double, Eigen::Dynamic, 1>& x, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& P)
{
  double now = getUnixTime();

  VectorXd q_old = x_old.block(0, 0, robot_->get_num_positions(), 1);

  // predict x to be within joint limits
  x = x_old;
  for (int i=0; i<q_old.rows(); i++){
    if (isfinite(robot_->joint_limit_min[i]) && x[i] < robot_->joint_limit_min[i]){
      x[i] = robot_->joint_limit_min[i];
    } else if (isfinite(robot_->joint_limit_max[i]) && x[i] > robot_->joint_limit_max[i]){
      x[i] = robot_->joint_limit_max[i];
    }
  }

  P = tracker->getCovariance();
  P += MatrixXd::Identity(P.rows(), P.cols())*0.00001; // avoid singularity

  /***********************************************
                DYNAMICS HINTS
    *********************************************/

  for (int i=1; i<robot_->bodies.size(); i++){
    // todo: some caching? maybe? this is pretty inefficient
    auto it = dynamics_vars_per_robot_.find(robot_->bodies[i]->get_model_name());
    DynamicsVars these_vars;
    if (it != dynamics_vars_per_robot_.end())
      these_vars = it->second;
    else
      these_vars = dynamics_vars_defaults_;

    if (robot_->bodies[i]->getJoint().is_floating())
      for (int i = 0; i < robot_->bodies[i]->getJoint().get_num_positions(); i++)
        P(i + robot_->bodies[i]->get_position_start_index(), i + robot_->bodies[i]->get_position_start_index()) += these_vars.floating_base_var;
    else
      for (int i = 0; i < robot_->bodies[i]->getJoint().get_num_positions(); i++)
        P(i + robot_->bodies[i]->get_position_start_index(), i + robot_->bodies[i]->get_position_start_index()) += these_vars.other_var;
  }

  if (verbose_)
    printf("Spent %f in dynamics constraints.\n", getUnixTime() - now);
  return true;
}
