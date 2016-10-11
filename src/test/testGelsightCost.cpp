/**
 *
 * This program tests the Gelsight cost independently by
 * spawning an object from a specified URDF and using it to
 * simulate Gelsight point clouds, which are fed into
 * a ManipulationTracker with just a Gelsight cost (and
 * a dynamics cost for smoothing.)
 * 
 * Both ground truth and estimated object pose are published
 * on state channels to be visualized via another process
 * (e.g. Director with both object and GT registered as
 * known manipulands via the directorconfig).
 *
 */

#include "ManipulationTracker.hpp"
#include "costs/DynamicsCost.hpp"
#include "costs/GelsightCost.hpp"
#include "yaml-cpp/yaml.h"
#include "common/common.hpp"
#include "GelsightOpenGLSim.hpp"

using namespace std;
using namespace Eigen;
using namespace drake::math;

void publish_gt_state(std::shared_ptr<lcm::LCM> lcm, std::shared_ptr<RigidBodyTree> robot, VectorXd x_robot, string gt_publish_channel){
  // publish GT
  bot_core::robot_state_t gt_state;
  gt_state.utime = getUnixTime();
  std::string robot_name = ""; // TODO(gizatt) does this need to be set?

  gt_state.num_joints = 0;
  bool found_floating = false;
  for (int i=1; i<robot->bodies.size(); i++){
    if (robot->bodies[i]->getJoint().isFloating()){
      Vector3d xyz = x_robot.block<3, 1>(robot->bodies[i]->get_position_start_index() + 0, 0);
      gt_state.pose.translation.x = xyz[0];
      gt_state.pose.translation.y = xyz[1];
      gt_state.pose.translation.z = xyz[2];
      auto quat = rpy2quat(x_robot.block<3, 1>(robot->bodies[i]->get_position_start_index() + 3, 0));
      gt_state.pose.rotation.w = quat.x(); // these two somehow disagree with each other... whyyy
      gt_state.pose.rotation.x = quat.y();
      gt_state.pose.rotation.y = quat.z();
      gt_state.pose.rotation.z = quat.w();
      if (found_floating){
        printf("Had more than one floating joint???\n");
        exit(-1);
      }
      found_floating = true;
    } else {
      // warning: if numpositions != numvelocities, problems arise...
      gt_state.num_joints += robot->bodies[i]->getJoint().getNumPositions();
      for (int j=0; j < robot->bodies[i]->getJoint().getNumPositions(); j++){
        gt_state.joint_name.push_back(robot->bodies[i]->getJoint().getPositionName(j));
        gt_state.joint_position.push_back(x_robot[robot->bodies[i]->get_position_start_index() + j]);
        gt_state.joint_velocity.push_back(x_robot[robot->bodies[i]->get_position_start_index() + j + robot->number_of_positions()]);
      }
    }
  }
  gt_state.joint_effort.resize(gt_state.num_joints, 0.0);
  lcm->publish(gt_publish_channel, &gt_state);
}

//estimator isn't estimating, unsure why
//closest point calls in drake are atrociously slow


Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> generateNewGelsightImage(
            GelsightOpenGLSim * glsim,
            std::shared_ptr<GelsightCost> gelsight_cost, 
            std::shared_ptr<RigidBodyTree> robot,
            const VectorXd x_robot){
  // this could be cleaner -- we could clean up the renderGelsight interface
  // but for now this'll make us robust to quat vs rpy parameterization
  VectorXd q_robot = x_robot.block(0, 0, robot->number_of_positions(), 1);
  auto robot_kinematics_cache = robot->doKinematics(q_robot);

  if (glsim){
    // get origin location and the "up" and "forward" vectors in body of interest
    // frame
    Matrix3Xd dirs(3, 3);
    dirs << 0, 0, 0, // origin 
            0, 0, 1, // fwd
            0, 1, 0; // up
    dirs = robot->transformPoints(robot_kinematics_cache, dirs, 0, 1);
    MatrixXd ret = glsim->renderGelsight(dirs.block<3, 1>(0, 0),
                                 dirs.block<3, 1>(0, 1),
                                 dirs.block<3, 1>(0, 2)-dirs.block<3, 1>(0, 0));
    // TODO(gizatt): actually pass in the gelsight sensor plane size too
    // and flip sign
    for (int i=0; i < ret.size(); i++)
      ret(i) = 1.0 - ret(i);
    return ret;
  } else {
    // fall back to drake raycasts
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> new_gelsight_image(gelsight_cost->get_input_num_pixel_rows(),
                                                                             gelsight_cost->get_input_num_pixel_cols());
    new_gelsight_image.setZero();
    
    int num_pixels = gelsight_cost->get_input_num_pixel_rows() * gelsight_cost->get_input_num_pixel_cols();
    
    // generate vector set for the raycast
    GelsightCost::SensorPlane sensor_plane = gelsight_cost->get_sensor_plane();
    Matrix3Xd origins(3, num_pixels);
    Matrix3Xd ray_endpoints(3, num_pixels);
    VectorXd distances(num_pixels);
    Matrix3Xd normals(3, num_pixels);
    
    for (int i=0; i < new_gelsight_image.rows(); i++){
      for (int j=0; j < new_gelsight_image.cols(); j++){
        int k = i*new_gelsight_image.cols() + j;
        Vector3d pt = sensor_plane.lower_left + 
            ((double)j)/((double)new_gelsight_image.cols())*(sensor_plane.lower_right - sensor_plane.lower_left) +
            ((double)i)/((double)new_gelsight_image.rows())*(sensor_plane.upper_left - sensor_plane.lower_left);
        origins.block<3, 1>(0, k) = pt - sensor_plane.normal * 1.0; // standoff for numerical improvement
        ray_endpoints.block<3, 1>(0, k) = pt + sensor_plane.normal * sensor_plane.thickness;
      }
    }

    robot->collisionRaycast(robot_kinematics_cache, origins, ray_endpoints, distances, normals, false);

    for (int i=0; i < new_gelsight_image.rows(); i++){
      for (int j=0; j < new_gelsight_image.cols(); j++){
        int k = i*new_gelsight_image.cols() + j;
        distances(k) -= 1.0;
        if (distances(k) >= 0)
          new_gelsight_image(i, j) = 1.0 - (distances(k) / sensor_plane.thickness);
        else
          new_gelsight_image(i, j) = 0.0;
      }
    }
    
    return new_gelsight_image;
  }
}

int main(int argc, char** argv) {
  const char* drc_path = std::getenv("DRC_BASE");
  if (!drc_path) {
    throw std::runtime_error("environment variable DRC_BASE is not set");
  }

  if (argc != 2){
    printf("Use: testGelsightCost <path to config file>\n");
    return 0;
  }


  std::shared_ptr<lcm::LCM> lcm(new lcm::LCM());
  if (!lcm->good()) {
    throw std::runtime_error("LCM is not good");
  }


  string configFile(argv[1]);
  YAML::Node config = YAML::LoadFile(configFile);

  // object model path
  GelsightOpenGLSim * glsim = NULL;
  if (config["obj"]){
    string object_model_path = config["obj"].as<string>();
    glsim = new GelsightOpenGLSim(object_model_path.c_str(), 0.001, 640, 480);
  }


  // gt publish channel
  string gt_publish_channel = config["gt_publish_channel"].as<string>();

  // get robot and any supplied initial condition
  string robot_urdf_path = string(drc_path) + config["urdf"].as<string>();
  shared_ptr<RigidBodyTree> robot(new RigidBodyTree(robot_urdf_path));
  VectorXd x0_robot(robot->number_of_positions() + robot->number_of_velocities());
  x0_robot.setZero();

  // collision check needs a non-const RBT... so generate another one
  std::shared_ptr<RigidBodyTree> robot_for_collision_sim(new RigidBodyTree(robot_urdf_path));

  // (we assemble this early as we need it for finding a good contact position)
  if (!config["gelsight_cost"]){
    printf("Config must specify a gelsight cost!\n");
    exit(-1);
  }
  // gelsight cost needs its own RBT to modify during collision checks
  std::shared_ptr<GelsightCost> gelsight_cost(new GelsightCost(shared_ptr<RigidBodyTree>(new RigidBodyTree(robot_urdf_path)), lcm, config["gelsight_cost"]));

  if (config["q0"] && config["q0"].Type() == YAML::NodeType::Map){
    for (int i=0; i < robot->number_of_positions(); i++){
      auto find = config["q0"][robot->getPositionName(i)];
      if (find){
        x0_robot(i) = find.as<double>();
      } 
      else // unnecessary but here for clarity
      {
        x0_robot(i) = 0.0;
      }
    }
  }
  //x0_robot(2) += 0.063 / cos(x0_robot(3)) / cos(x0_robot(4));
  


  // gradually bring it down until the minimum point on the simulated depth return
  // is sufficiently small
  int k = 0;
  
  ManipulationTracker estimator_init(robot, x0_robot, lcm, config, true);
  estimator_init.addCost(dynamic_pointer_cast<ManipulationTrackerCost, GelsightCost>(gelsight_cost));

  while (1){
    publish_gt_state(lcm, robot, x0_robot, gt_publish_channel);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> new_gelsight_image = generateNewGelsightImage(glsim, gelsight_cost,
          robot_for_collision_sim, x0_robot);

    gelsight_cost->updateGelsightImage(new_gelsight_image);

    estimator_init.update();
    estimator_init.publish();

    if (new_gelsight_image.maxCoeff() > 0.8)
      break;

    if (k > 10000){
      printf("Couldn't get a good contact...\n");
      exit(-1);
    }

    x0_robot(2) -= 0.0005;
  }

  VectorXd x0_err(robot->number_of_positions() + robot->number_of_velocities());
  x0_err.setZero();
  if (config["q0_err"] && config["q0_err"].Type() == YAML::NodeType::Map){
    for (int i=0; i < robot->number_of_positions(); i++){
      auto find = config["q0_err"][robot->getPositionName(i)];
      if (find)
        x0_err(i) = find.as<double>();
      else // unnecessary but here for clarity
        x0_err(i) = 0.0;
    }
  }
  

  VectorXd x_robot_est = x0_robot + x0_err; // estimator init
  VectorXd x_robot = x0_robot; // ground truth
  
  // initialize tracker itself
  ManipulationTracker estimator(robot, x_robot_est, lcm, config, true);
  
  // finally register the gelsight cost we generated earlier
  estimator.addCost(dynamic_pointer_cast<ManipulationTrackerCost, GelsightCost>(gelsight_cost));

  std::cout << "Gelsight Test Starting" << std::endl;

  double t = 0.0;
  double timestep = 0.01;

  if (config["timestep"])
    timestep = config["timestep"].as<double>();

  while(1){
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> new_gelsight_image = generateNewGelsightImage(glsim, gelsight_cost,
      robot_for_collision_sim, x_robot);

    gelsight_cost->updateGelsightImage(new_gelsight_image);

    estimator.update();
    estimator.publish();

    publish_gt_state(lcm, robot, x_robot, gt_publish_channel);
    
    usleep(10); // yield for other procs to avoid breaking my OS
  }

  std::cout << "Gelsight Test Ending" << std::endl;
  return 0;
}