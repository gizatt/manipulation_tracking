#undef NDEBUG
#include <assert.h> 
#include <fstream>
#include "DynamicsCost.hpp"
#include "drake/util/convexHull.h"
#include "zlib.h"
#include <cmath>
#include "common.hpp"

using namespace std;
using namespace Eigen;

DynamicsCost::DynamicsCost(std::shared_ptr<const RigidBodyTree> robot_, std::shared_ptr<lcm::LCM> lcm_, YAML::Node config) :
    robot(robot_),
    lcm(lcm_),
    nq(robot->num_positions)
{
  if (config["dynamics_floating_base_var"])
    dynamics_floating_base_var = config["dynamics_floating_base_var"].as<double>();
  if (config["dynamics_other_var"])
    dynamics_other_var = config["dynamics_other_var"].as<double>();
  if (config["joint_limit_var"])
    joint_limit_var = config["joint_limit_var"].as<double>();
  if (config["verbose"])
    verbose = config["verbose"].as<bool>();
}

/***********************************************
            KNOWN POSITION HINTS
*********************************************/
bool DynamicsCost::constructCost(ManipulationTracker * tracker, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Q, Eigen::Matrix<double, Eigen::Dynamic, 1>& f, double& K)
{
  double now = getUnixTime();

  VectorXd x_old = tracker->output();
  VectorXd q_old = x_old.block(0, 0, robot->num_positions, 1);

  /***********************************************
                DYNAMICS HINTS
    *********************************************/
  if (!std::isinf(dynamics_other_var) || !std::isinf(dynamics_floating_base_var)){
    double DYNAMICS_OTHER_WEIGHT = std::isinf(dynamics_other_var) ? 0.0 : 1. / (2. * dynamics_other_var * dynamics_other_var);
    double DYNAMICS_FLOATING_BASE_WEIGHT = std::isinf(dynamics_floating_base_var) ? 0.0 : 1. / (2. * dynamics_floating_base_var * dynamics_floating_base_var);
    
    double now_dyn = getUnixTime();
    // for now, zoh on dynamics
    // min (x - x')^2
    // i.e. min x^2 - 2xx' + x'^2
    for (int i=0; i<6; i++){
      Q(i, i) += DYNAMICS_FLOATING_BASE_WEIGHT*1.0;
      f(i) -= DYNAMICS_FLOATING_BASE_WEIGHT*q_old(i);
      K += DYNAMICS_FLOATING_BASE_WEIGHT*q_old(i)*q_old(i);
    }
    for (int i=6; i<q_old.rows(); i++){
      Q(i, i) += DYNAMICS_OTHER_WEIGHT*1.0;
      f(i) -= DYNAMICS_OTHER_WEIGHT*q_old(i);
      K += DYNAMICS_OTHER_WEIGHT*q_old(i)*q_old(i);
    }
    if (verbose)
      printf("Spent %f in joint known weight constraints.\n", getUnixTime() - now_dyn);
  }
    
  /***********************************************
                JOINT LIMIT CONSTRAINTS
    *********************************************/
  if (!std::isinf(joint_limit_var)){
    double JOINT_LIMIT_WEIGHT = 1. / (2. * joint_limit_var * joint_limit_var);

    double now_lim = getUnixTime();
    // push negative ones back towards their limits
    // phi_jl(i) = J_jl(i,i)*(x - lim)
    // (back out lim = x - phi_jl(i)/J_jl(i,i)
    // min phi_li^2 if phi_jl < 0, so
    // min (J_jl(i,i)*(x-lim))^2
    // min x^2 - 2 * lim * x + lim^2
    for (int i=0; i<q_old.rows(); i++){
      if (isfinite(robot->joint_limit_min[i]) && q_old[i] < robot->joint_limit_min[i]){
        Q(i, i) += JOINT_LIMIT_WEIGHT*1.0;
        f(i) -= JOINT_LIMIT_WEIGHT*robot->joint_limit_min[i];
        K += JOINT_LIMIT_WEIGHT*robot->joint_limit_min[i]*robot->joint_limit_min[i];
      }
      if (isfinite(robot->joint_limit_max[i]) && q_old[i] > robot->joint_limit_max[i]){
        Q(i, i) += JOINT_LIMIT_WEIGHT*1.0;
        f(i) -= JOINT_LIMIT_WEIGHT*robot->joint_limit_max[i];
        K += JOINT_LIMIT_WEIGHT*robot->joint_limit_max[i]*robot->joint_limit_max[i];
      }
    }
    if (verbose)
      printf("Spent %f in joint limit constraints.\n", getUnixTime() - now_lim);
  }


  if (verbose)
    printf("Spent %f in dynamics constraints.\n", getUnixTime() - now);
  return true;
}