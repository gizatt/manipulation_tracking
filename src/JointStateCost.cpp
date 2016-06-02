#undef NDEBUG
#include <assert.h> 
#include <fstream>
#include "JointStateCost.hpp"
#include "drake/util/convexHull.h"
#include "zlib.h"
#include <cmath>
#include "common.hpp"

using namespace std;
using namespace Eigen;

JointStateCost::JointStateCost(std::shared_ptr<const RigidBodyTree> robot_, std::shared_ptr<lcm::LCM> lcm_, YAML::Node config) :
    robot(robot_),
    lcm(lcm_),
    nq(robot->number_of_positions())
{
  if (config["state_channelname"])
    state_channelname = config["state_channelname"].as<string>();
  if (config["joint_reported_var"])
    joint_reported_var = config["joint_reported_var"].as<double>();
  if (config["timeout_time"])
    timeout_time = config["timeout_time"].as<double>();
  if (config["verbose"])
    verbose = config["verbose"].as<bool>();
  if (config["listen_joints"])
    for (auto iter=config["listen_joints"].begin(); iter!=config["listen_joints"].end(); iter++)
      listen_joints.push_back((*iter).as<string>());

  if (state_channelname.size() > 0){
    state_sub = lcm->subscribe(state_channelname, &JointStateCost::handleJointStateMsg, this);
    state_sub->setQueueCapacity(1);
  } else {
    printf("JointStateCost was not handed a state_channelname, so I'm not doing anything!\n");
  }

  lastReceivedTime = getUnixTime() - timeout_time*2.;
  
  x_robot_measured.resize(nq);
  x_robot_measured_known.resize(nq);
}

/***********************************************
            KNOWN POSITION HINTS
*********************************************/
bool JointStateCost::constructCost(ManipulationTracker * tracker, const Eigen::Matrix<double, Eigen::Dynamic, 1> x_old, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Q, Eigen::Matrix<double, Eigen::Dynamic, 1>& f, double& K)
{
  double now = getUnixTime();
  if (now - lastReceivedTime > timeout_time){
    if (verbose)
      printf("JointStateCost: constructed but timed out\n");
    return false;
  }
  else {
    now = getUnixTime();

    double JOINT_REPORTED_WEIGHT = std::isinf(joint_reported_var) ? 0.0 : 1. / (2. * joint_reported_var * joint_reported_var);

    // copy over last known info
    x_robot_measured_mutex.lock();
    VectorXd q_measured = x_robot_measured.block(0,0,nq,1);
    std::vector<bool> x_robot_measured_known_copy = x_robot_measured_known;
    x_robot_measured_mutex.unlock();

    // min (x - x')^2
    // i.e. min x^2 - 2xx' + x'^2

    for (int i=0; i<nq; i++){
      if (x_robot_measured_known_copy[i]){
        Q(i, i) += JOINT_REPORTED_WEIGHT*1.0;
        f(i) -= JOINT_REPORTED_WEIGHT*q_measured(i);
        K += JOINT_REPORTED_WEIGHT*q_measured(i)*q_measured(i);
      }
    }
    if (verbose)
      printf("Spent %f in joint reported state constraints, channel %s.\n", getUnixTime() - now, state_channelname.c_str());
    return true;
  }
}

void JointStateCost::handleJointStateMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::joint_state_t* msg){
  //printf("Received hand state on channel  %s\n", chan.c_str());
  x_robot_measured_mutex.lock();

  map<string, int> map = robot->computePositionNameToIndexMap();
  for (int i=0; i < msg->num_joints; i++){
    auto id = map.end();
    // only use this one if either we have no listen joints set
    // or if this is one of them
    if (listen_joints.size() == 0 || find(listen_joints.begin(), listen_joints.end(), msg->joint_name[i])!=listen_joints.end()){
      id = map.find(msg->joint_name[i]);
    }

    if (id != map.end()){
      x_robot_measured(id->second) = msg->joint_position[i];
      x_robot_measured_known[id->second] = true;
    }
  }

  x_robot_measured_mutex.unlock();
}