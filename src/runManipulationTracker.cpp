
#include "ManipulationTracker.hpp"
#include "yaml-cpp/yaml.h"

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
  const char* drc_path = std::getenv("DRC_BASE");
  if (!drc_path) {
    throw std::runtime_error("environment variable DRC_BASE is not set");
  }

  if (argc != 2){
    printf("Use: irb140_runEstimator <path to yaml config file>\n");
    return 0;
  }

  string configFile(argv[1]);
  YAML::Node config = YAML::LoadFile(configFile);

  std::shared_ptr<RigidBodyTree> arm(new RigidBodyTree(std::string(drc_path) + config["manipulator"]["urdf"].as<string>()));
  arm->compile();  
  VectorXd x0_arm(arm->num_positions + arm->num_velocities);
  x0_arm*=0;


  // just box and table
  /*
  std::shared_ptr<RigidBodyTree> manipuland(new RigidBodyTree(std::string(drc_path) + "/software/control/src/jasmine_tea_box.urdf"));
  VectorXd x0_manipuland = VectorXd::Zero(manipuland->num_positions + manipuland->num_velocities);
  x0_manipuland.block<6, 1>(0, 0) << 0.5, 0.0, 0.88, 0.0, 0.0, 1.5;
  x0_manipuland.block<6, 1>(6, 0) << 0.5, 0.0, 0.7, 0.0, 0.0, 0.0;
  */

  // just hand
  /*
  std::shared_ptr<RigidBodyTree> manipuland(new RigidBodyTree(std::string(drc_path) + "/software/control/src/urdf/robotiq_simple_collision.urdf"));
  VectorXd x0_manipuland = VectorXd::Zero(manipuland->num_positions + manipuland->num_velocities);
  x0_manipuland.block<6, 1>(0, 0) << 0.5, 0.0, 1.21, 1.6, -1.14, -3.36;
  */

  // hand and box and table
  /*
  std::shared_ptr<RigidBodyTree> manipuland(new RigidBodyTree(std::string(drc_path) + "/software/control/src/jasmine_tea_box.urdf"));
  manipuland->addRobotFromURDF(std::string(drc_path) + "/software/drake/drake/examples/Atlas/urdf/robotiq_simple.urdf", DrakeJoint::ROLLPITCHYAW);
  VectorXd x0_manipuland = VectorXd::Zero(manipuland->num_positions + manipuland->num_velocities);
  x0_manipuland.block<6, 1>(0, 0) << 0.5, 0.0, 0.88, 0.0, 0.0, 1.5;
  x0_manipuland.block<6, 1>(6, 0) << 0.5, 0.0, 0.7, 0.0, 0.0, 0.0;
  x0_manipuland.block<6, 1>(12, 0) << 0.5, 0.0, 1.21, 1.6, -1.14, -3.36;
  */

  // arm and box and table
/*  
  std::shared_ptr<RigidBodyTree> manipuland(new RigidBodyTree(std::string(drc_path) + "/software/control/src/urdf/irb140_chull_robotiq_actuated_fingers.urdf"));
  manipuland->addRobotFromURDF(std::string(drc_path) + "/software/control/src/jasmine_tea_box.urdf", DrakeJoint::ROLLPITCHYAW);
  manipuland->addRobotFromURDF(std::string(drc_path) + "/software/control/src/desk.urdf", DrakeJoint::ROLLPITCHYAW);
  VectorXd x0_manipuland = VectorXd::Zero(manipuland->num_positions + manipuland->num_velocities);
  x0_manipuland.block<6, 1>(0, 0) << -.17, 0.0, .91, 0.0, 0.0, 0.0;
  x0_manipuland.block<6, 1>(manipuland->num_positions-12, 0) << 0.67, 0.0, 0.8, 0.0, 0.0, 0.0;
  x0_manipuland.block<6, 1>(manipuland->num_positions-6, 0) << 0.5, 0.0, 0.7, 0.0, 0.0, 0.0;
  manipuland->compile();
  */

/*
  // arm and table and cardboard box with lid
  std::shared_ptr<RigidBodyTree> manipuland(new RigidBodyTree(std::string(drc_path) + "/software/control/src/urdf/irb140_chull_robotiq_actuated_fingers.urdf"));
  manipuland->addRobotFromURDF(std::string(drc_path) + "/software/control/src/desk.urdf", DrakeJoint::ROLLPITCHYAW);
  manipuland->addRobotFromURDF(std::string(drc_path) + "/software/control/src/urdf/cardbox_box_hollow_with_lid.urdf", DrakeJoint::ROLLPITCHYAW);
  VectorXd x0_manipuland = VectorXd::Zero(manipuland->num_positions + manipuland->num_velocities);
  x0_manipuland.block<6, 1>(0, 0) << -.17, 0.0, .91, 0.0, 0.0, 0.0;
  x0_manipuland.block<6, 1>(manipuland->num_positions-16, 0) << 0.67, 0.0, 0.45, 0.0, 0.0, 0.0;
  x0_manipuland.block<6, 1>(manipuland->num_positions-10, 0) << 0.67, 0.0, 0.71, 0.0, 0.0, 0.0, -0.8, -0.8, -0.8, -0.8;
  manipuland->compile();
*/

  // generate manipuland from yaml file

  int old_num_positions = 0;
  auto manip = config["manipulands"].begin();
  std::shared_ptr<RigidBodyTree> manipuland(new RigidBodyTree(std::string(drc_path) + manip->second["urdf"].as<string>()));

  VectorXd q0_manipuland = VectorXd::Zero(manipuland->num_positions);
  if (manip->second["q0"] && manip->second["q0"].Type() == YAML::NodeType::Map){
    for (int i=old_num_positions; i < manipuland->num_positions; i++){
      auto find = manip->second["q0"][manipuland->getPositionName(i)];
      if (find)
        q0_manipuland(i) = find.as<double>();
      else // unnecessary in this first loop but here for clarity
        q0_manipuland(i) = 0.0;
    }
  }

  old_num_positions = manipuland->num_positions;
  manip++;
  while (manip != config["manipulands"].end()){
    manipuland->addRobotFromURDF(std::string(drc_path) + manip->second["urdf"].as<string>(), DrakeJoint::ROLLPITCHYAW);
    q0_manipuland.conservativeResize(manipuland->num_positions);
    if (manip->second["q0"] && manip->second["q0"].Type() == YAML::NodeType::Map){
      for (int i=old_num_positions; i < manipuland->num_positions; i++){
        auto find = manip->second["q0"][manipuland->getPositionName(i)];
        if (find){
          q0_manipuland(i) = find.as<double>();
        }
        else{
          q0_manipuland(i) = 0.0;
        }
      }
    } else {
      q0_manipuland.block(old_num_positions, 0, manipuland->num_positions-old_num_positions, 1) *= 0.0;
    }

    old_num_positions = manipuland->num_positions;
    manip++;
  }
  manipuland->compile();

  cout << "All position names and init values: " << endl;
  for (int i=0; i < manipuland->num_positions; i++){
    cout << "\t " << i << ": \"" << manipuland->getPositionName(i) << "\" = " << q0_manipuland[i] << endl;
  }

  VectorXd x0_manipuland = VectorXd::Zero(manipuland->num_positions + manipuland->num_velocities);
  x0_manipuland.block(0,0,manipuland->num_positions,1) = q0_manipuland;

  std::unique_ptr<IRB140Estimator> estimator(new IRB140Estimator(arm, manipuland, x0_arm, x0_manipuland,
    (std::string(drc_path) + config["config"].as<string>()).c_str(), config["manipulator"]["state_channel"].as<string>().c_str(), true, "ROBOTIQ_LEFT_STATE"));

  std::cout << "IRB140 Estimator Listening" << std::endl;
  estimator->run();
  return 0;
}