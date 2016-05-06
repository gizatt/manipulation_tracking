
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
    printf("Use: runManipulationTracker <path to yaml config file>\n");
    return 0;
  }

  string configFile(argv[1]);
  YAML::Node config = YAML::LoadFile(configFile);

  std::shared_ptr<RigidBodyTree> arm(new RigidBodyTree(std::string(drc_path) + config["manipulator"]["urdf"].as<string>()));
  arm->compile();  
  VectorXd x0_arm(arm->num_positions + arm->num_velocities);
  x0_arm*=0;

  // generate manipuland from yaml file by adding each robot in sequence
  // first robot -- need to initialize the RBT
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
  // each new robot can be added via addRobotFromURDF
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

  // tracker itself
  std::unique_ptr<ManipulationTracker> estimator(new ManipulationTracker(arm, manipuland, x0_arm, x0_manipuland,
    (std::string(drc_path) + config["config"].as<string>()).c_str(), config["manipulator"]["state_channel"].as<string>().c_str(), true, "ROBOTIQ_LEFT_STATE"));

  // if the yaml file has bounds, set them in the estimator
  if (config["bounds"]){
    ManipulationTracker::BoundingBox bounds;
    bounds.xMin = config["bounds"]["x"][0].as<double>();
    bounds.xMax = config["bounds"]["x"][1].as<double>();
    bounds.yMin = config["bounds"]["y"][0].as<double>();
    bounds.yMax = config["bounds"]["y"][1].as<double>();
    bounds.zMin = config["bounds"]["z"][0].as<double>();
    bounds.zMax = config["bounds"]["z"][1].as<double>();
    estimator->setBounds(bounds);
    cout << "Bounds: " << endl;
    cout << "\t" << bounds.xMin << ", " << bounds.xMax << endl;
    cout << "\t" << bounds.yMin << ", " << bounds.yMax << endl;
    cout << "\t" << bounds.zMin << ", " << bounds.zMax << endl;
  } else {
    cout << "Bounds not set" << endl;
  }

  // if the yaml file has cost info, set them in the estimator
  if (config["costs"]){
    ManipulationTracker::CostInfoStore cost_info;
    if (config["costs"]["icp_kinect"]){
      cost_info.icp_var = config["costs"]["icp_kinect"]["icp_var"].as<double>();
      cost_info.MAX_CONSIDERED_ICP_DISTANCE = config["costs"]["icp_kinect"]["max_considered_icp_distance"].as<double>();
      cost_info.MIN_CONSIDERED_JOINT_DISTANCE = config["costs"]["icp_kinect"]["min_considered_joint_distance"].as<double>();
    }
    if (config["costs"]["free_space"]){
      cost_info.free_space_var = config["costs"]["free_space"]["free_space_var"].as<double>();
    }
    if (config["costs"]["known_position"]){
      cost_info.joint_known_fb_var = config["costs"]["known_position"]["joint_known_fb_var"].as<double>();
      cost_info.joint_known_encoder_var = config["costs"]["known_position"]["joint_known_encoder_var"].as<double>();
    }
    if (config["costs"]["joint_limit"]){
      cost_info.joint_limit_var = config["costs"]["joint_limit"]["joint_limit_var"].as<double>();
    }
    if (config["costs"]["position_constraint"]){
      cost_info.position_constraint_var = config["costs"]["position_constraint"]["position_constraint_var"].as<double>();
    }
    if (config["costs"]["dynamics"]){
      cost_info.dynamics_floating_base_var = config["costs"]["dynamics"]["dynamics_floating_base_var"].as<double>();
      cost_info.dynamics_other_var = config["costs"]["dynamics"]["dynamics_other_var"].as<double>();
    }
    estimator->setCosts(cost_info);
    cout << "Costs loaded." << endl;
  } else {
    cout << "Costs not loaded." << endl;
  }

  std::cout << "Manipulation Tracker Listening" << std::endl;
  estimator->run();
  return 0;
}