
#include <assert.h> 
#include <fstream>
#include "drake/util/convexHull.h"
#include <cmath>
#include <cfloat>
#include "lcmtypes/bot_core/robot_state_t.hpp"
#include "lcmtypes/bot_core/rigid_transform_t.hpp"
#include "common.hpp"
#include "drake/systems/plants/RigidBodyTree.h"
#include "ManipulationTrackerCost.hpp"
#include "yaml-cpp/yaml.h"
#include <lcm/lcm-cpp.hpp>
#include <stdexcept>
#include <iostream>
#include <mutex>



using namespace std;
using namespace Eigen;

RigidBodyTree * robot_1;
RigidBodyTree * robot_2;


class Handler
{
  private:
    bool q_valid = false;
    VectorXd q;
    mutex state_mutex;
  public:
    Handler(shared_ptr<lcm::LCM> lcm, string channel) {
      auto state_sub = lcm->subscribe(channel, &Handler::handleStateRobot, this);
      state_sub->setQueueCapacity(1);
    }

    ~Handler() {}

    void handleStateRobot(const lcm::ReceiveBuffer* rbuf,
                             const std::string& chan,
                             const bot_core::robot_state_t* msg){
      state_mutex.lock();
      q.resize(msg->num_joints + 7);
      q[0] = msg->pose.translation.x;
      q[1] = msg->pose.translation.y;
      q[2] = msg->pose.translation.z;
      q[3] = msg->pose.rotation.w;
      q[4] = msg->pose.rotation.x;
      q[5] = msg->pose.rotation.y;
      q[6] = msg->pose.rotation.z;

      for (int i=0; i < msg->num_joints; i++){
        q[7+i] = msg->joint_position[i];
      }
      q_valid = true;
      state_mutex.unlock();
    }

    bool get_q_valid() {
      return q_valid;
    }

    VectorXd get_q() {
      VectorXd ret;
      state_mutex.lock();
      ret = q;
      state_mutex.unlock();
      return ret;
    }
};

void broadcast_transform(shared_ptr<lcm::LCM> lcm, string channel, Isometry3d transform){
  bot_core::rigid_transform_t tf_msg;
  tf_msg.utime = getUnixTime()*1000*1000;
  tf_msg.trans[0] = transform.matrix()(0,3);
  tf_msg.trans[1] = transform.matrix()(1,3);
  tf_msg.trans[2] = transform.matrix()(2,3);

  Quaterniond quat(transform.rotation());
  tf_msg.quat[0] = quat.w();
  tf_msg.quat[1] = quat.x();
  tf_msg.quat[2] = quat.y();
  tf_msg.quat[3] = quat.z();

  lcm->publish(channel, &tf_msg);
}

Isometry3d getTransform(VectorXd q_r1, int link_ind_1, Vector3d offset_1, VectorXd q_r2, int link_ind_2, Vector3d offset_2, Isometry3d extra_transform){
  Isometry3d transform;
  transform.setIdentity();

  KinematicsCache<double> robot_kinematics_cache_1(robot_1->bodies);
  KinematicsCache<double> robot_kinematics_cache_2(robot_2->bodies);

  robot_kinematics_cache_1.initialize(q_r1);
  robot_1->doKinematics(robot_kinematics_cache_1);
  robot_kinematics_cache_2.initialize(q_r2);
  robot_2->doKinematics(robot_kinematics_cache_2);

  Isometry3d tf_offset_1;
  tf_offset_1.setIdentity();
  tf_offset_1.matrix().block<3, 1>(0, 3) = offset_1;


  Isometry3d tf_offset_2;
  tf_offset_2.setIdentity();
  tf_offset_2.matrix().block<3, 1>(0, 3) = offset_2;

  // do forwardkin to the frame of choice
  Isometry3d transform_1 = robot_1->relativeTransform(robot_kinematics_cache_1, 0, link_ind_1) * tf_offset_1.inverse();

  Isometry3d transform_2_pre = robot_2->relativeTransform(robot_kinematics_cache_2, 0, link_ind_2);
  transform_2_pre.matrix().block<3, 3>(0, 0) *= extra_transform.matrix().block<3, 3>(0, 0);
  Isometry3d transform_2 = transform_2_pre * tf_offset_2.inverse();

  return transform_2.inverse() * transform_1;
}

int main(int argc, char** argv) {
  const char* drc_path = std::getenv("DRC_BASE");
  if (!drc_path) {
    throw std::runtime_error("environment variable DRC_BASE is not set");
  }

  if (argc != 2){
    printf("Use: runManipulationTrackerIRB140 <path to yaml config file>\n");
    return 0;
  }

  printf("WARNING: IF ERROR IS LARGER THAN YOU EXPECT, CHECK ROTATIONAL"
          "SYMMETRIES THAT MIGHT CONFUSE TRACKER.\n");

  std::shared_ptr<lcm::LCM> lcm(new lcm::LCM());
  if (!lcm->good()) {
    throw std::runtime_error("LCM is not good");
  }

  string configFile(argv[1]);
  YAML::Node config = YAML::LoadFile(configFile);
  
  Isometry3d relative_transform;
  relative_transform.setIdentity();
  if (config["relative_transform"]){
    vector<double> relative_transform_vec = config["relative_transform"].as<vector<double>>();
    relative_transform.matrix().block<3, 1>(0,3) = Vector3d(relative_transform_vec[0], relative_transform_vec[1], relative_transform_vec[2]);
    auto relative_transform_quat = drake::math::rpy2quat(Vector3d(relative_transform_vec[3], relative_transform_vec[4], relative_transform_vec[5])*3.141592/180.);
    relative_transform.matrix().block<3, 3>(0,0) = Quaterniond(relative_transform_quat[0], relative_transform_quat[1],relative_transform_quat[2], relative_transform_quat[3]).toRotationMatrix();
  }

  robot_1 = new RigidBodyTree(drc_path + config["robot_1"]["urdf"].as<string>(), DrakeJoint::QUATERNION);
  string link_1_name = config["robot_1"]["link"].as<string>();
  int link_ind_1 = robot_1->FindBodyIndex(link_1_name);
  Vector3d offset_1;
  offset_1.setZero();
  if (config["robot_1"]["offset"]){
    std::vector<double> offset = config["robot_1"]["offset"].as<std::vector<double>>();
    offset_1[0] = offset[0];
    offset_1[1] = offset[1];
    offset_1[2] = offset[2];
  }

  robot_2 = new RigidBodyTree(drc_path + config["robot_2"]["urdf"].as<string>(), DrakeJoint::QUATERNION);
  string link_2_name = config["robot_2"]["link"].as<string>();
  int link_ind_2 = robot_2->FindBodyIndex(link_2_name);
  Vector3d offset_2;
  offset_2.setZero();
  if (config["robot_2"]["offset"]){
    std::vector<double> offset = config["robot_2"]["offset"].as<std::vector<double>>();
    offset_2[0] = offset[0];
    offset_2[1] = offset[1];
    offset_2[2] = offset[2];
  }

  Handler handlerRobot1_tracking(lcm, config["robot_1"]["channel"].as<string>());
  Handler handlerRobot1_gt(lcm, config["robot_1"]["gt_channel"].as<string>());
  Handler handlerRobot2_tracking(lcm, config["robot_2"]["channel"].as<string>());
  Handler handlerRobot2_gt(lcm, config["robot_2"]["gt_channel"].as<string>());

  std::cout << "Listening...\n" << endl;
  while(1){
      usleep(1000*10);
      lcm->handleTimeout(0.01);

      if (handlerRobot1_tracking.get_q_valid() &&
          handlerRobot1_gt.get_q_valid() &&
          handlerRobot2_tracking.get_q_valid() &&
          handlerRobot2_gt.get_q_valid()){

        VectorXd robot_1_q = handlerRobot1_tracking.get_q();
        VectorXd robot_1_q_gt = handlerRobot1_gt.get_q();
        VectorXd robot_2_q = handlerRobot2_tracking.get_q();
        VectorXd robot_2_q_gt = handlerRobot2_gt.get_q();

        Isometry3d transform_identity; transform_identity.setIdentity();
        Isometry3d transform_gt = getTransform(robot_1_q_gt, link_ind_1, offset_1, robot_2_q_gt, link_ind_2, offset_2, transform_identity);
        Isometry3d transform = getTransform(robot_1_q, link_ind_1, offset_1, robot_2_q, link_ind_2, offset_2, relative_transform);

        cout << "*****" << endl;
        cout << "GT: " << transform_gt.matrix() << endl;
        cout << "Est: " << transform.matrix() << endl;
        cout << "Diff: " << (transform * transform_gt.inverse()).matrix() << endl;
        cout << "*****\n\n\n" << endl;

        broadcast_transform(lcm, "ERROR_TRANSFORM_GT", transform_gt);
        broadcast_transform(lcm, "ERROR_TRANSFORM_EST", transform);
        broadcast_transform(lcm, "ERROR_TRANSFORM_ERR", (transform * transform_gt.inverse()));
      }
  }

  return 0;
}