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
    nq(robot->number_of_positions())
{
  if (config["dynamics_floating_base_var"])
    dynamics_floating_base_var = config["dynamics_floating_base_var"].as<double>();
  if (config["dynamics_other_var"])
    dynamics_other_var = config["dynamics_other_var"].as<double>();
  if (config["verbose"])
    verbose = config["verbose"].as<bool>();
}

/***********************************************
            KNOWN POSITION HINTS
*********************************************/
bool DynamicsCost::constructPredictionMatrices(ManipulationTracker * tracker, const Eigen::Matrix<double, Eigen::Dynamic, 1> x_old, Eigen::Matrix<double, Eigen::Dynamic, 1>& x, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& P)
{
  double now = getUnixTime();

  VectorXd q_old = x_old.block(0, 0, robot->number_of_positions(), 1);

  // predict x to be within joint limits
  x = x_old;
  for (int i=0; i<q_old.rows(); i++){
    if (isfinite(robot->joint_limit_min[i]) && x[i] < robot->joint_limit_min[i]){
      x[i] = robot->joint_limit_min[i];
    } else if (isfinite(robot->joint_limit_max[i]) && x[i] > robot->joint_limit_max[i]){
      x[i] = robot->joint_limit_max[i];
    }
  }

  P = tracker->getCovariance();
  P += MatrixXd::Identity(P.rows(), P.cols())*0.00001; // avoid singularity

  /***********************************************
                DYNAMICS HINTS
    *********************************************/
  if (!std::isinf(dynamics_other_var)){
    for (int i=0; i < 6; i++){
      P(i, i) += dynamics_floating_base_var;
    }
  }
  if (!std::isinf(dynamics_floating_base_var)){
    for (int i=6; i<x_old.rows(); i++){
      P(i, i) += dynamics_other_var;
    }
  }

  if (verbose)
    printf("Spent %f in dynamics constraints.\n", getUnixTime() - now);
  return true;
}