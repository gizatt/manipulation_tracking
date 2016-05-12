#undef NDEBUG
#include <assert.h> 
#include <fstream>
#include "ManipulationTracker.hpp"
#include "drake/util/convexHull.h"
#include "zlib.h"
#include "sdf_2d_functions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <cfloat>
#include "drake/systems/plants/joints/RevoluteJoint.h"
#include "lcmtypes/bot_core/robot_state_t.hpp"
#include "common.hpp"

using namespace std;
using namespace Eigen;

std::shared_ptr<RigidBodyTree> setupRobotFromConfig(YAML::Node config, Eigen::VectorXd& x0_robot, std::string base_path, bool verbose){
  // generate robot from yaml file by adding each robot in sequence
  // first robot -- need to initialize the RBT
  int old_num_positions = 0;
  auto manip = config["robots"].begin();
  std::shared_ptr<RigidBodyTree> robot(new RigidBodyTree(base_path + manip->second["urdf"].as<string>()));
  x0_robot.resize(robot->num_positions);
  if (manip->second["q0"] && manip->second["q0"].Type() == YAML::NodeType::Map){
    for (int i=old_num_positions; i < robot->num_positions; i++){
      auto find = manip->second["q0"][robot->getPositionName(i)];
      if (find)
        x0_robot(i) = find.as<double>();
      else // unnecessary in this first loop but here for clarity
        x0_robot(i) = 0.0;
    }
  }
  old_num_positions = robot->num_positions;
  manip++;
  // each new robot can be added via addRobotFromURDF
  while (manip != config["robots"].end()){
    robot->addRobotFromURDF(base_path + manip->second["urdf"].as<string>(), DrakeJoint::ROLLPITCHYAW);
    x0_robot.conservativeResize(robot->num_positions);
    if (manip->second["q0"] && manip->second["q0"].Type() == YAML::NodeType::Map){
      for (int i=old_num_positions; i < robot->num_positions; i++){
        auto find = manip->second["q0"][robot->getPositionName(i)];
        if (find){
          x0_robot(i) = find.as<double>();
        }
        else{
          x0_robot(i) = 0.0;
        }
      }
    } else {
      x0_robot.block(old_num_positions, 0, robot->num_positions-old_num_positions, 1) *= 0.0;
    }

    old_num_positions = robot->num_positions;
    manip++;
  }
  robot->compile();

  x0_robot.conservativeResize(robot->num_positions + robot->num_velocities);

  if (verbose){
    cout << "All position names and init values: " << endl;
    for (int i=0; i < robot->num_positions; i++){
      cout << "\t " << i << ": \"" << robot->getPositionName(i) << "\" = " << x0_robot[i] << endl;
    }
  }
  return robot;
}


ManipulationTracker::ManipulationTracker(std::shared_ptr<RigidBodyTree> robot_, Eigen::Matrix<double, Eigen::Dynamic, 1> x0_robot_, std::shared_ptr<lcm::LCM> lcm_, bool verbose_) :
    robot(robot_),
    lcm(lcm_),
    x_robot(x0_robot_),
    verbose(verbose_),
    robot_kinematics_cache(robot->bodies)
{
  if (robot->num_positions + robot->num_velocities != x_robot.rows()){
    printf("Expected initial condition with %d rows, got %ld rows.\n", robot->num_positions + robot->num_velocities, x_robot.rows());
    exit(0);
  }
}

void ManipulationTracker::update(){

  // set up a quadratic program:
  // 0.5 * x.' Q x + f.' x
  // and since we're unconstrained then solve as linear system
  // Qx = -f

  VectorXd q_old = x_robot.block(0, 0, robot->num_positions, 1);
  int nq = robot->num_positions;
  robot_kinematics_cache.initialize(q_old);
  robot->doKinematics(robot_kinematics_cache);
  
  double now=getUnixTime();

  // set up a quadratic program:
  // 0.5 * x.' Q x + f.' x
  // and since we're unconstrained then solve as linear system
  // Qx = -f

  VectorXd f(nq);
  f.setZero();
  MatrixXd Q(nq, nq);
  Q.setZero();
  double K = 0.;
  // generate from registered costs:
  for (auto it=registeredCosts.begin(); it != registeredCosts.end(); it++){
    VectorXd f_new(nq);
    f_new.setZero();
    MatrixXd Q_new(nq, nq);
    Q_new.setZero();
    double K_new = 0.0;
    bool use = (*it)->constructCost(this, Q_new, f_new, K_new);
    if (use){
      f += f_new;
      Q += Q_new;
      K += K_new;
    }
  }
  // solve if we had any useful costs:
  if (K > 0.0){
    // cut out variables that do not enter at all -- i.e., their row and column of Q, and row of f, are 0
    MatrixXd Q_reduced;
    VectorXd f_reduced;

    // is it used?
    std::vector<bool> rows_used(nq, false);
    int nq_reduced = 0;
    for (int i=0; i < nq; i++){
      if ( !(fabs(f[i]) <= 1E-10 && Q.block(i, 0, 1, nq).norm() <= 1E-10 && Q.block(0, i, nq, 1).norm() <= 1E-10) ){
        rows_used[i] = true;
        nq_reduced++;
      }
    }
    // do this reduction (collapse the rows/cols of vars that don't correspond)
    Q_reduced.resize(nq_reduced, nq_reduced);
    f_reduced.resize(nq_reduced, 1);
    int ir = 0, jr = 0;
    for (int i=0; i < nq; i++){
      if (rows_used[i]){
        jr = 0;
        for (int j=0; j < nq; j++){
          if (rows_used[j]){
            Q_reduced(ir, jr) = Q(i, j);
            jr++;
          }
        }
        f_reduced[ir] = f[i];
        ir++;
      }
    }

    // perform reduced solve
    VectorXd q_new_reduced = Q_reduced.colPivHouseholderQr().solve(-f_reduced);

    // reexpand
    ir = 0;
    for (int i=0; i < nq; i++){
      if (rows_used[i] && q_new_reduced[ir] == q_new_reduced[ir]){
        x_robot[i] = q_new_reduced[ir];
        ir++;
      }
    }
  }

  if (verbose)
    printf("Total elapsed in least squares construction and solve: %f\n", getUnixTime() - now);

} 

void ManipulationTracker::publish(){
  // Publish the object state
  //cout << "robot robot name vector: " << robot->robot_name.size() << endl;
  for (int roboti=1; roboti < robot->robot_name.size(); roboti++){
    bot_core::robot_state_t manipulation_state;
    manipulation_state.utime = getUnixTime();
    std::string robot_name = robot->robot_name[roboti];

    manipulation_state.num_joints = 0;
    bool found_floating = false;
    for (int i=0; i<robot->bodies.size(); i++){
      if (robot->bodies[i]->model_name == robot_name){
        if (robot->bodies[i]->getJoint().isFloating()){
          manipulation_state.pose.translation.x = x_robot[robot->bodies[i]->position_num_start + 0];
          manipulation_state.pose.translation.y = x_robot[robot->bodies[i]->position_num_start + 1];
          manipulation_state.pose.translation.z = x_robot[robot->bodies[i]->position_num_start + 2];
          auto quat = rpy2quat(x_robot.block<3, 1>(robot->bodies[i]->position_num_start + 3, 0));
          manipulation_state.pose.rotation.w = quat[0];
          manipulation_state.pose.rotation.x = quat[1];
          manipulation_state.pose.rotation.y = quat[2];
          manipulation_state.pose.rotation.z = quat[3];
          if (found_floating){
            printf("Had more than one floating joint???\n");
            exit(-1);
          }
          found_floating = true;
        } else {
          // warning: if numpositions != numvelocities, problems arise...
          manipulation_state.num_joints += robot->bodies[i]->getJoint().getNumPositions();
          for (int j=0; j < robot->bodies[i]->getJoint().getNumPositions(); j++){
            manipulation_state.joint_name.push_back(robot->bodies[i]->getJoint().getPositionName(j));
            manipulation_state.joint_position.push_back(x_robot[robot->bodies[i]->position_num_start + j]);
            manipulation_state.joint_velocity.push_back(x_robot[robot->bodies[i]->position_num_start + j + robot->num_positions]);
          }
        }
      }
    }
    manipulation_state.joint_effort.resize(manipulation_state.num_joints, 0.0);
    std::string channelname = "EST_MANIPULAND_STATE_" + robot_name;
    lcm->publish(channelname, &manipulation_state);
  }
}