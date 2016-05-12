#undef NDEBUG
#include <assert.h> 
#include <fstream>
#include "RobotStateCost.hpp"
#include "drake/util/convexHull.h"
#include "zlib.h"
#include <cmath>
#include "common.hpp"

using namespace std;
using namespace Eigen;

RobotStateCost::RobotStateCost(std::shared_ptr<RigidBodyTree> robot_, std::shared_ptr<lcm::LCM> lcm_, YAML::Node config) :
    robot(robot_),
    lcm(lcm_),
    nq(robot->num_positions)
{
  if (config["state_channelname"])
    state_channelname = config["state_channelname"].as<string>();
  if (config["joint_known_encoder_var"])
    joint_known_encoder_var = config["joint_known_encoder_var"].as<double>();
  if (config["joint_known_floating_base_var"])
    joint_known_floating_base_var = config["joint_known_floating_base_var"].as<double>();
  if (config["timeout_time"])
    timeout_time = config["timeout_time"].as<double>();
  if (config["verbose"])
    verbose = config["verbose"].as<bool>();
  if (config["transcribe_floating_base_vars"])
    transcribe_floating_base_vars = config["transcribe_floating_base_vars"].as<bool>();

  state_sub = lcm->subscribe(state_channelname, &RobotStateCost::handleRobotStateMsg, this);
  state_sub->setQueueCapacity(1);
  lastReceivedTime = getUnixTime() - timeout_time*2.;
  
  x_robot_measured.resize(nq);
  x_robot_measured_known.resize(nq);
}

/***********************************************
            KNOWN POSITION HINTS
*********************************************/
bool RobotStateCost::constructCost(ManipulationTracker * tracker, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Q, Eigen::Matrix<double, Eigen::Dynamic, 1>& f, double& K)
{
  double now = getUnixTime();
  if (now - lastReceivedTime > timeout_time){
    if (verbose)
      printf("RobotStateCost: constructed but timed out\n");
    return false;
  }
  else {
    now = getUnixTime();

    double JOINT_KNOWN_ENCODER_WEIGHT = std::isinf(joint_known_encoder_var) ? 0.0 : 1. / (2. * joint_known_encoder_var * joint_known_encoder_var);
    double JOINT_KNOWN_FLOATING_BASE_WEIGHT = std::isinf(joint_known_floating_base_var) ? 0.0 : 1. / (2. * joint_known_floating_base_var * joint_known_floating_base_var);

    // copy over last known info
    x_robot_measured_mutex.lock();
    VectorXd q_measured = x_robot_measured.block(0,0,nq,1);
    std::vector<bool> x_robot_measured_known_copy = x_robot_measured_known;
    x_robot_measured_mutex.unlock();

    // min (x - x')^2
    // i.e. min x^2 - 2xx' + x'^2

    for (int i=0; i<6; i++){
      if (x_robot_measured_known_copy[i]){
        Q(i, i) += JOINT_KNOWN_FLOATING_BASE_WEIGHT*1.0;
        f(i) -= JOINT_KNOWN_FLOATING_BASE_WEIGHT*q_measured(i);
        K += JOINT_KNOWN_FLOATING_BASE_WEIGHT*q_measured(i)*q_measured(i);
      }
    }
    for (int i=6; i<nq; i++){
      if (x_robot_measured_known_copy[i]){
        Q(i, i) += JOINT_KNOWN_ENCODER_WEIGHT*1.0;
        f(i) -= JOINT_KNOWN_ENCODER_WEIGHT*q_measured(i);
        K += JOINT_KNOWN_ENCODER_WEIGHT*q_measured(i)*q_measured(i);
      }
    }
    if (verbose)
      printf("Spent %f in robot reported state constraints, channel %s\n", getUnixTime() - now, state_channelname.c_str());
    return true;
  }
}

void RobotStateCost::handleRobotStateMsg(const lcm::ReceiveBuffer* rbuf,
                         const std::string& chan,
                         const bot_core::robot_state_t* msg){
  if (verbose)
    printf("Received robot state on channel  %s\n", chan.c_str());
  lastReceivedTime = getUnixTime();

  x_robot_measured_mutex.lock();

  if (transcribe_floating_base_vars){
    x_robot_measured(0) = msg->pose.translation.x;
    x_robot_measured(1) = msg->pose.translation.y;
    x_robot_measured(2) = msg->pose.translation.z;

    auto quat = Quaterniond(msg->pose.rotation.w, msg->pose.rotation.x, msg->pose.rotation.y, msg->pose.rotation.z);
    x_robot_measured.block<3, 1>(3, 0) = quat.toRotationMatrix().eulerAngles(2, 1, 0);
    for (int i=0; i < 6; i++)
      x_robot_measured_known[i] = true;
  }
  

  map<string, int> map = robot->computePositionNameToIndexMap();
  for (int i=0; i < msg->num_joints; i++){
    auto id = map.find(msg->joint_name[i]);
    if (id != map.end()){
      x_robot_measured(id->second) = msg->joint_position[i];
      x_robot_measured_known[id->second] = true;
    }
  }

  x_robot_measured_mutex.unlock();

}
