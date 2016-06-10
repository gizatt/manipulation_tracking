
#include <assert.h> 
#include <fstream>
#include "drake/util/convexHull.h"
#include "zlib.h"
#include <cmath>
#include <cfloat>
#include "lcmtypes/bot_core/robot_state_t.hpp"
#include "common.hpp"
#include "drake/systems/plants/RigidBodyTree.h"
#include "ManipulationTrackerCost.hpp"
#include "yaml-cpp/yaml.h"
#include <lcm/lcm-cpp.hpp>
#include <stdexcept>
#include <iostream>


using namespace std;
using namespace Eigen;

bool q_1_valid = false;
VectorXd q_1;
bool q_2_valid = false;
VectorXd q_2;
RigidBodyTree * robot_1;
RigidBodyTree * robot_2;

class Handler
{
  public:
    ~Handler() {}

    void handleStateRobot1(const lcm::ReceiveBuffer* rbuf,
                             const std::string& chan,
                             const bot_core::robot_state_t* msg){
      q_1.resize(msg->num_joints + 7);
      q_1[0] = msg->pose.translation.x;
      q_1[1] = msg->pose.translation.y;
      q_1[2] = msg->pose.translation.z;
      q_1[3] = msg->pose.rotation.w;
      q_1[4] = msg->pose.rotation.x;
      q_1[5] = msg->pose.rotation.y;
      q_1[6] = msg->pose.rotation.z;

      for (int i=0; i < msg->num_joints; i++){
        q_1[7+i] = msg->joint_position[i];
      }
    q_1_valid = true;
    }

    void handleStateRobot2(const lcm::ReceiveBuffer* rbuf,
                               const std::string& chan,
                               const bot_core::robot_state_t* msg){
      q_2.resize(msg->num_joints + 7);
      q_2[0] = msg->pose.translation.x;
      q_2[1] = msg->pose.translation.y;
      q_2[2] = msg->pose.translation.z;
      q_2[3] = msg->pose.rotation.w;
      q_2[4] = msg->pose.rotation.x;
      q_2[5] = msg->pose.rotation.y;
      q_2[6] = msg->pose.rotation.z;

      for (int i=0; i < msg->num_joints; i++){
        q_2[7+i] = msg->joint_position[i];
      }
      q_2_valid = true;
    }

};



int main(int argc, char** argv) {
  const char* drc_path = std::getenv("DRC_BASE");
  if (!drc_path) {
    throw std::runtime_error("environment variable DRC_BASE is not set");
  }

  if (argc != 2){
    printf("Use: runManipulationTrackerIRB140 <path to yaml config file>\n");
    return 0;
  }


  std::shared_ptr<lcm::LCM> lcm(new lcm::LCM());
  if (!lcm->good()) {
    throw std::runtime_error("LCM is not good");
  }

  string configFile(argv[1]);
  YAML::Node config = YAML::LoadFile(configFile);

  string output_channel = config["output_channel"].as<string>();
  
  robot_1 = new RigidBodyTree(drc_path + config["robot_1"]["urdf"].as<string>(), DrakeJoint::QUATERNION);
  string channel_1 = config["robot_1"]["channel"].as<string>();
  string link_1_name = config["robot_1"]["link"].as<string>();
  int link_ind_1 = robot_1->findLinkId(link_1_name);


  robot_2 = new RigidBodyTree(drc_path + config["robot_2"]["urdf"].as<string>(), DrakeJoint::QUATERNION);
  string channel_2 = config["robot_2"]["channel"].as<string>();
  string link_2_name = config["robot_2"]["link"].as<string>();
  int link_ind_2 = robot_2->findLinkId(link_2_name);

  KinematicsCache<double> robot_kinematics_cache_1(robot_1->bodies);
  KinematicsCache<double> robot_kinematics_cache_2(robot_2->bodies);

  Handler handlerObject;
  auto robot_1_state_sub = lcm->subscribe(channel_1, &Handler::handleStateRobot1, &handlerObject);
  auto robot_2_state_sub = lcm->subscribe(channel_2, &Handler::handleStateRobot2, &handlerObject);

  robot_1_state_sub->setQueueCapacity(1);
  robot_2_state_sub->setQueueCapacity(1);

  std::cout << "Listening...\n" << endl;
  while(1){
      lcm->handleTimeout(0);

      if (q_1_valid && q_2_valid){

        // do kinematics for both
        robot_kinematics_cache_1.initialize(q_1);
        robot_1->doKinematics(robot_kinematics_cache_1);
        robot_kinematics_cache_2.initialize(q_2);
        robot_2->doKinematics(robot_kinematics_cache_2);

        // do forwardkin to the frame of choice
        Isometry3d transform_1 = robot_1->relativeTransform(robot_kinematics_cache_1, 1, link_ind_1);
        Isometry3d transform_2 = robot_2->relativeTransform(robot_kinematics_cache_2, 1, link_ind_2);
        cout << (transform_2.inverse() * transform_1).matrix() << endl;
      }
  }

  return 0;
}