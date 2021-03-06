#undef NDEBUG
#include <assert.h> 
#include <fstream>
#include "ManipulationTracker.hpp"
#include <cmath>
#include <cfloat>
#include "drake/util/drakeGeometryUtil.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/multibody/parsers/urdf_parser.h"

#include "common/common.hpp"

using namespace std;
using namespace Eigen;
using namespace drake::math;
using namespace drake::parsers::urdf;

std::shared_ptr<RigidBodyTree<double>> setupRobotFromConfig(YAML::Node config, Eigen::VectorXd& x0_robot, std::string base_path, bool verbose, bool less_collision){
  // generate robot from yaml file by adding each robot in sequence
  // first robot -- need to initialize the RBT
  int old_num_positions = 0;
  
  string robots_string;
  if (less_collision)
    robots_string = "robots_less_collision";
  else
    robots_string = "robots";

  auto manip = config[robots_string].begin();
  std::shared_ptr<RigidBodyTree<double>> robot(new RigidBodyTree<double>());
  AddModelInstanceFromUrdfFileWithRpyJointToWorld(base_path + manip->second["urdf"].as<string>(), robot.get());

  x0_robot.resize(robot->get_num_positions());
  if (manip->second["q0"] && manip->second["q0"].Type() == YAML::NodeType::Map){
    for (int i=old_num_positions; i < robot->get_num_positions(); i++){
      auto find = manip->second["q0"][robot->get_position_name(i)];
      if (find)
        x0_robot(i) = find.as<double>();
      else // unnecessary in this first loop but here for clarity
        x0_robot(i) = 0.0;
    }
  }
  old_num_positions = robot->get_num_positions();
  manip++;
  // each new robot can be added via addRobotFromURDF
  while (manip != config[robots_string].end()){
    AddModelInstanceFromUrdfFileWithRpyJointToWorld(base_path + manip->second["urdf"].as<string>(), robot.get());
    x0_robot.conservativeResize(robot->get_num_positions());
    if (manip->second["q0"] && manip->second["q0"].Type() == YAML::NodeType::Map){
      for (int i=old_num_positions; i < robot->get_num_positions(); i++){
        auto find = manip->second["q0"][robot->get_position_name(i)];
        if (find){
          x0_robot(i) = find.as<double>();
        }
        else{
          x0_robot(i) = 0.0;
        }
      }
    } else {
      x0_robot.block(old_num_positions, 0, robot->get_num_positions()-old_num_positions, 1) *= 0.0;
    }

    old_num_positions = robot->get_num_positions();
    manip++;
  }
  robot->compile();

  x0_robot.conservativeResize(robot->get_num_positions() + robot->get_num_velocities());
  x0_robot.block(robot->get_num_positions(), 0, robot->get_num_velocities(), 1).setZero();

  if (verbose){
    cout << "All position names and init values: " << endl;
    for (int i=0; i < robot->get_num_positions(); i++){
      cout << "\t " << i << ": \"" << robot->get_position_name(i) << "\" = " << x0_robot[i] << endl;
    }
  }
  return robot;
}

static bool vector_contains_str(std::vector<std::string> vec, std::string str) {
  for (int i=0; i<vec.size(); i++) {
    if (str == vec[i])
      return true;
  }
  return false;
}

std::shared_ptr<RigidBodyTree<double>> setupRobotFromConfigSubset(YAML::Node config, Eigen::VectorXd& x0_robot_subset, std::string base_path,
    bool verbose, bool less_collision, bool exclusionary, std::vector<std::string> exceptions, std::vector<int> &index_correspondences){
  // generate robot from yaml file by adding each robot in sequence
  // if some robots are to be excluded, their yaml file names will appear in exceptions
  // first robot -- need to initialize the RBT
  int old_num_positions = 0;
  int old_num_positions_subset = 0;
  
  string robots_string;
  if (less_collision)
    robots_string = "robots_less_collision";
  else
    robots_string = "robots";

  auto manip = config[robots_string].begin();
  
  std::shared_ptr<RigidBodyTree<double>> robot(new RigidBodyTree<double>());
  std::shared_ptr<RigidBodyTree<double>> robot_subset(new RigidBodyTree<double>());

  x0_robot_subset.resize(robot_subset->get_num_positions());

  old_num_positions = robot->get_num_positions();
  old_num_positions_subset = robot_subset->get_num_positions();

  // each new robot can b
  while (manip != config[robots_string].end()) {
    AddModelInstanceFromUrdfFileWithRpyJointToWorld(base_path + manip->second["urdf"].as<string>(), robot.get());
    if (vector_contains_str(exceptions, manip->first.as<std::string>()) != exclusionary) { // check (CONTAINED) XOR (EXCLUSIONARY)
      AddModelInstanceFromUrdfFileWithRpyJointToWorld(base_path + manip->second["urdf"].as<string>(), robot_subset.get());

      x0_robot_subset.conservativeResize(robot_subset->get_num_positions());

      if (manip->second["q0"] && manip->second["q0"].Type() == YAML::NodeType::Map){
        for (int i=old_num_positions_subset; i < robot_subset->get_num_positions(); i++){
          auto find = manip->second["q0"][robot_subset->get_position_name(i)];
          if (find){
            x0_robot_subset(i) = find.as<double>();
          }
          else{
            x0_robot_subset(i) = 0.0;
          }
        }
      } else {
        x0_robot_subset.block(old_num_positions_subset, 0, robot_subset->get_num_positions()-old_num_positions_subset, 1) *= 0.0;
      }

      int cur_position = old_num_positions;
      for (int i=old_num_positions_subset; i < robot_subset->get_num_positions(); i++){
        // for each new index, make a correspondence with whole robot's indices
        index_correspondences.push_back(cur_position);
        cur_position++;
      }
    }

    old_num_positions = robot->get_num_positions();
    old_num_positions_subset = robot_subset->get_num_positions();

    manip++;
  }
  robot_subset->compile();

//  x0_robot_subset.conservativeResize(robot_subset->get_num_positions() + robot_subset->get_num_velocities());
//  x0_robot_subset.block(robot_subset->get_num_positions(), 0, robot->get_num_velocities(), 1).setZero();

  if (verbose){
    cout << "All position names and init values: " << endl;
    for (int i=0; i < robot_subset->get_num_positions(); i++){
      cout << "\t " << i << ": \"" << robot->get_position_name(i) << "\" = " << x0_robot_subset[i] << endl;
    }
  }

  return robot_subset;
}

ManipulationTracker::ManipulationTracker(std::shared_ptr<const RigidBodyTree<double>> robot, Eigen::Matrix<double, Eigen::Dynamic, 1> x0_robot, std::shared_ptr<lcm::LCM> lcm, YAML::Node config, bool verbose) :
    robot_(robot),
    lcm_(lcm),
    verbose_(verbose),
    robot_kinematics_cache_(robot->doKinematics(x0_robot))
{
  if (robot_->get_num_positions() + robot_->get_num_velocities() != x0_robot.rows()){
    printf("Expected initial condition with %d rows, got %ld rows.\n", robot_->get_num_positions() + robot_->get_num_velocities(), x0_robot.rows());
    exit(0);
  }

  // spawn initial decision variables from robot state
  x_.resize(x0_robot.rows());
  x_.setZero();
  x_.block(0,0,x0_robot.rows(), 1) = x0_robot;
  covar_.resize(x0_robot.rows(), x0_robot.rows());
  covar_.setZero(); // TODO: how to better initialize covariance?
  covar_ += MatrixXd::Identity(covar_.rows(), covar_.cols())*0.0000000001;

  // generate robot names
  for (auto it=robot->bodies.begin(); it!=robot->bodies.end(); it++){
    auto findit = find(robot_names_.begin(), robot_names_.end(), (*it)->get_model_name());
    if (findit == robot_names_.end())
      robot_names_.push_back((*it)->get_model_name());
  }

  // get dynamics configuration from yaml
  if (config["dynamics"]){
    if (config["dynamics"]["dynamics_floating_base_var"])
      dynamics_vars_defaults_.floating_base_var = config["dynamics"]["dynamics_floating_base_var"].as<double>();
    if (config["dynamics"]["dynamics_other_var"])
      dynamics_vars_defaults_.other_var = config["dynamics"]["dynamics_other_var"].as<double>();
    if (config["dynamics"]["verbose"])
      dynamics_verbose_ = config["dynamics"]["verbose"].as<bool>();

    if (config["dynamics"]["robot_specific_vars"]){
      for (auto iter = config["dynamics"]["robot_specific_vars"].begin();
                iter != config["dynamics"]["robot_specific_vars"].end();
                iter++){
        DynamicsVars robot_vars;
        robot_vars.floating_base_var = (*iter)["dynamics_floating_base_var"].as<double>();
        robot_vars.other_var = (*iter)["dynamics_other_var"].as<double>();
        dynamics_vars_per_robot_[(*iter)["robot"].as<string>()] = robot_vars;
        printf("Loaded vars %f, %f for robot %s\n", robot_vars.floating_base_var, robot_vars.other_var, (*iter)["robot"].as<string>().c_str());
      }
    }
  }

  // get publish info from yaml
  if (config["publish"]){
    for (auto manip = config["publish"].begin(); manip != config["publish"].end(); manip++)
    {
      publish_info new_publish_info;
      new_publish_info.robot_name = manip->second["robot_name"].as<string>();
      new_publish_info.publish_type = manip->second["type"].as<string>();
      new_publish_info.publish_channel = manip->second["channel"].as<string>();
      publish_infos_.push_back(new_publish_info);
    }
  }

  // do we need to rectify the estimated state by transforming to a new frame?
  // puts base of source_robot into dest_frame, and moves all floating bases
  // by that same transform
  if (config["post_transform"]){
    do_post_transform_ = true;
    post_transform_robot_ = config["post_transform"]["source_robot"].as<string>();
    post_transform_dest_frame_ = config["post_transform"]["dest_frame"].as<string>(); 
  }

  if (config["force_align"]){
    do_force_align_ = true;
    force_align_robot_ = config["force_align"]["source_robot"].as<string>();
    force_align_dest_frame_ = config["force_align"]["dest_frame"].as<string>(); 
  }


  const char * filename = NULL;
  if (config["bot_param_filename"])
    filename = config["bot_param_filename"].as<string>().c_str();
  this->initBotConfig(filename);

}


void ManipulationTracker::initBotConfig(const char* filename)
{
  if (filename && filename[0])
    {
      string filename_full = string(std::getenv("DRC_BASE")) + string(filename);
      botparam_ = bot_param_new_from_file(filename_full.c_str());
    }
  else
    {
    while (!botparam_)
      {
        botparam_ = bot_param_new_from_server(lcm_->getUnderlyingLCM(), 0);
      }
    }
  botframes_ = bot_frames_get_global(lcm_->getUnderlyingLCM(), botparam_);
}

int ManipulationTracker::get_trans_with_utime(std::string from_frame, std::string to_frame,
                               long long utime, Eigen::Isometry3d & mat)
{
  if (!botframes_)
  {
    std::cout << "botframe is not initialized" << std::endl;
    mat = mat.matrix().Identity();
    return 0;
  }

  int status;
  double matx[16];
  status = bot_frames_get_trans_mat_4x4_with_utime( botframes_, from_frame.c_str(),  to_frame.c_str(), utime, matx);
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      mat(i,j) = matx[i*4+j];
    }
  }
  return status;
}

void ManipulationTracker::addCost(std::shared_ptr<ManipulationTrackerCost> new_cost){
  // spawn any necessary new decision variables
  int new_decision_vars = new_cost->getNumExtraVars();
  if (new_decision_vars > 0){
    x_.conservativeResize(x_.rows() + new_decision_vars);
    x_.block(x_.rows()-new_decision_vars, 0, new_decision_vars, 1) = new_cost->getExtraVarsX0();
    covar_.conservativeResize(x_.rows(), x_.rows());
    covar_.block(covar_.rows()-new_decision_vars, 0, new_decision_vars, covar_.cols()).setZero();
    covar_.block(0, covar_.cols()-new_decision_vars, covar_.rows(), new_decision_vars).setZero();
    covar_.block(covar_.rows()-new_decision_vars, covar_.cols()-new_decision_vars, new_decision_vars, new_decision_vars) = MatrixXd::Identity(new_decision_vars, new_decision_vars)*0.000001;
  }
  CostAndView new_cost_and_view;
  new_cost_and_view.first = new_cost;

  // robot positions and velocities are the first nX indices
  for (int i=0; i < robot_->get_num_positions() + robot_->get_num_velocities(); i++){
    new_cost_and_view.second.push_back(i);
  }
  // the rest of the new vars are on the very end
  for (int i= (x_.rows() - new_decision_vars); i < x_.rows(); i++){
    new_cost_and_view.second.push_back(i);
  }

  registeredCostInfo_.push_back(new_cost_and_view);
}


void ManipulationTracker::update(){

  // Performs an EKF-like update of our given state.

  // get the robot state ready:
  int nq = robot_->get_num_positions();
  int nx = x_.rows();
  VectorXd q_old = x_.block(0, 0, nq, 1);
  robot_kinematics_cache_.initialize(q_old);
  robot_->doKinematics(robot_kinematics_cache_);
  
  double now=getUnixTime();

  // PREDICTION:
  // Do a standard EKF prediction step: update the state 
  // via potentially nonlinear dynamics functions,
  // and update covariance via their jacobian.
  VectorXd x_pred(x_);
  MatrixXd covar_pred(covar_);

  // predict x to be within joint limits
  for (int i=0; i<q_old.rows(); i++){
    if (isfinite(robot_->joint_limit_min[i]) && x_pred[i] < robot_->joint_limit_min[i]){
      x_pred[i] = robot_->joint_limit_min[i];
    } else if (isfinite(robot_->joint_limit_max[i]) && x_pred[i] > robot_->joint_limit_max[i]){
      x_pred[i] = robot_->joint_limit_max[i];
    }
  }

  for (int i=1; i<robot_->bodies.size(); i++){
    // todo: some caching? maybe? this is pretty inefficient
    auto it = dynamics_vars_per_robot_.find(robot_->bodies[i]->get_model_name());
    DynamicsVars these_vars;
    if (it != dynamics_vars_per_robot_.end())
      these_vars = it->second;
    else
      these_vars = dynamics_vars_defaults_;

    printf("For robot %s and link %s, using vars %f and %f\n", robot_->bodies[i]->get_model_name().c_str(), robot_->bodies[i]->get_name().c_str(), these_vars.floating_base_var, these_vars.other_var);
    if (robot_->bodies[i]->getJoint().is_floating() && !std::isinf(these_vars.floating_base_var)){
      for (int j = 0; j < robot_->bodies[i]->getJoint().get_num_positions(); j++){
        covar_pred(j + robot_->bodies[i]->get_position_start_index(), j + robot_->bodies[i]->get_position_start_index()) += these_vars.floating_base_var;
      }
      for (int j = 0; j < robot_->bodies[i]->getJoint().get_num_velocities(); j++){
        covar_pred(j + robot_->bodies[i]->get_velocity_start_index(), j + robot_->bodies[i]->get_velocity_start_index()) += these_vars.floating_base_var;
      }
    }
    else if (!robot_->bodies[i]->getJoint().is_floating() && !std::isinf(these_vars.other_var)){
      for (int j = 0; j < robot_->bodies[i]->getJoint().get_num_positions(); j++){
        covar_pred(j + robot_->bodies[i]->get_position_start_index(), j + robot_->bodies[i]->get_position_start_index()) += these_vars.other_var;
      }
      for (int j = 0; j < robot_->bodies[i]->getJoint().get_num_velocities(); j++){
        covar_pred(j + robot_->bodies[i]->get_velocity_start_index(), j + robot_->bodies[i]->get_velocity_start_index()) += these_vars.other_var;
      }
    }
  }
  cout << "X Pred: " << x_pred.transpose() << endl;
  cout << "Covar Pred: " << covar_pred.diagonal().transpose() << endl;

  // TODO: accept parameters for this or pawn it off to a 
  // different object

  // MEASUREMENT:
  // Following the DART folks, we'll set this up as an
  // optimization instead of a linear update.
  // In our case, we'll set up a quadratic program -- many of these
  // costs are easy to write and balance in this way as 
  // instantaneous maximum-likelihood estimates, particularly
  // point cloud measurement errors.

  // So, for now, the optimization problem is:
  // 0.5 * x.' Q x + f.' x
  // and since we're unconstrained then solve as linear system
  // Qx = -f
  // With covariance estimate coming in from the Hessian Q

  VectorXd f(nx);
  f.setZero();
  MatrixXd Q(nx, nx);
  Q.setZero();
  double K = 0.;

  // generate these from registered costs:
  for (auto it=registeredCostInfo_.begin(); it != registeredCostInfo_.end(); it++){
    int nx_this = (*it).second.size();
    VectorXd f_new(nx_this);
    f_new.setZero();
    MatrixXd Q_new(nx_this, nx_this);
    Q_new.setZero();
    double K_new = 0.0;

    VectorXd x_old(nx_this);
    for (int i=0; i < nx_this; i++){
      x_old[i] = x_[(*it).second[i]];
    }
    bool use = (*it).first->constructCost(this, x_old, Q_new, f_new, K_new);
    if (use){
      for (int i=0; i<nx_this; i++){
        int loc_i = (*it).second[i];
        f(loc_i) += f_new(i);
        for (int j=0; j<nx_this; j++){
          int loc_j = (*it).second[j];
          Q(loc_i, loc_j) += Q_new(i, j);
        }
      }
      K += K_new;
    }
  }

  // and, following DART folks specifically here, include the prediction step
  // as an additional cost by penalizing squared Mahalanobis distance
  // (x - x_pred)' (covar_pred^(-1)) (x - x_pred)
  // which can be rewritten in our form as 
  covar_pred = covar_pred.inverse();
  Q += covar_pred;
  f -= x_pred.transpose() * covar_pred;
  K += 2 * x_pred.transpose() * covar_pred * x_pred;

  // do measurement update proper if we had any useful costs
  if (fabs(K) > 0.0){
    // cut out variables that do not enter at all -- i.e., their row and column of Q, and row of f, are 0
    MatrixXd Q_reduced;
    VectorXd f_reduced;

    // is it used?
    std::vector<bool> rows_used(nx, false);
    int nx_reduced = 0;
    for (int i=0; i < nx; i++){
      if ( !(fabs(f[i]) <= 1E-10 && Q.block(i, 0, 1, nx).norm() <= 1E-10 && Q.block(0, i, nx, 1).norm() <= 1E-10) ){
        rows_used[i] = true;
        nx_reduced++;
      }
    }
    // do this reduction (collapse the rows/cols of vars that don't correspond)
    Q_reduced.resize(nx_reduced, nx_reduced);
    f_reduced.resize(nx_reduced, 1);
    int ir = 0, jr = 0;
    for (int i=0; i < nx; i++){
      if (rows_used[i]){
        jr = 0;
        for (int j=0; j < nx; j++){
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
    ColPivHouseholderQR<MatrixXd> QR = Q_reduced.colPivHouseholderQr();
    VectorXd q_new_reduced = QR.solve(-f_reduced);
    MatrixXd Q_reduced_inverse = Q_reduced.inverse();

    // reexpand
    ir = 0;
    for (int i=0; i < nx; i++){
      if (rows_used[i] && q_new_reduced[ir] == q_new_reduced[ir]){
        // update of mean:
        x_[i] = q_new_reduced[ir];

        // update of covar for this row:
        int jr = 0;
        for (int j=0; j < nx; j++){
          if (rows_used[j] && q_new_reduced[jr] == q_new_reduced[jr]){
            covar_(i, j) = Q_reduced_inverse(ir, jr);
            jr++;
          }
        }

        ir++;
      }
    }
  }

  if (do_force_align_){
    Isometry3d force_align;
    force_align.setIdentity();
    // find the floating base of the desired robot
    int floating_base_body = -1;
    for (int i=0; i<robot_->bodies.size(); i++){
      if (robot_->bodies[i]->get_model_name() == force_align_robot_ && robot_->bodies[i]->getJoint().is_floating()){
        floating_base_body = i;
        break;
      }
    }
    if (floating_base_body < 0){
      printf("couldn't find desired floating base for post transform!\n");
      exit(1);
    }

    auto quat = rpy2quat(x_.block<3, 1>(robot_->bodies[floating_base_body]->get_position_start_index() + 3, 0));
    force_align.matrix().block<3, 3>(0,0) = Quaterniond(quat[0], quat[1], quat[2], quat[3]).matrix();
    force_align.matrix().block<3, 1>(0,3) = x_.block(robot_->bodies[floating_base_body]->get_position_start_index(), 0, 3, 1);

    // find error between that and the designated frame
    long long utime = 0;
    Eigen::Isometry3d to_dest_frame;
    this->get_trans_with_utime(force_align_dest_frame_, "local", utime, to_dest_frame);

    /*
    cout << "to robot base: " << force_align.matrix() << endl;
    cout << "dest_frame: " << to_dest_frame.matrix() << endl;
    cout << "post: " << force_align.matrix() << endl;
    */
    force_align =  to_dest_frame * force_align.inverse();
    // and update all floating bases in our robot with that
    for (int i=1; i<robot_->bodies.size(); i++){
      if (robot_->bodies[i]->getJoint().is_floating()){
        x_.block<3, 1>(robot_->bodies[i]->get_position_start_index() + 0, 0) = 
          force_align*x_.block<3, 1>(robot_->bodies[i]->get_position_start_index() + 0, 0);

        Quaterniond quat1(force_align.rotation());
        auto quat2 = rpy2quat(x_.block<3, 1>(robot_->bodies[i]->get_position_start_index() + 3, 0));
        quat1 *= Quaterniond(quat2[0], quat2[1], quat2[2], quat2[3]);
        x_.block<3,1>(robot_->bodies[i]->get_position_start_index() + 3, 0) = quat2rpy(Vector4d(quat1.w(), quat1.x(), quat1.y(), quat1.z()));
      }
    }
  }

  if (verbose_)
    printf("Total elapsed in EKF update: %f\n", getUnixTime() - now);

} 

void ManipulationTracker::publish(){
  cout << "State: " << x_.transpose() << endl;
  
  Isometry3d post_transform;
  post_transform.setIdentity();
  if (do_post_transform_){
    // find the floating base of the desired robot
    int floating_base_body = -1;
    for (int i=0; i<robot_->bodies.size(); i++){
      if (robot_->bodies[i]->get_model_name() == post_transform_robot_ && robot_->bodies[i]->getJoint().is_floating()){
        floating_base_body = i;
        break;
      }
    }
    if (floating_base_body < 0){
      printf("couldn't find desired floating base for post transform!\n");
      exit(1);
    }

    auto quat = rpy2quat(x_.block<3, 1>(robot_->bodies[floating_base_body]->get_position_start_index() + 3, 0));
    post_transform.matrix().block<3, 3>(0,0) = Quaterniond(quat[0], quat[1], quat[2], quat[3]).matrix();
    post_transform.matrix().block<3, 1>(0,3) = x_.block(robot_->bodies[floating_base_body]->get_position_start_index(), 0, 3, 1);

    // find error between that and the designated frame
    long long utime = 0;
    Eigen::Isometry3d to_dest_frame;
    this->get_trans_with_utime(post_transform_dest_frame_, "local", utime, to_dest_frame);

    cout << "to robot base: " << post_transform.matrix() << endl;
    cout << "dest_frame: " << to_dest_frame.matrix() << endl;
    post_transform =  to_dest_frame * post_transform.inverse();
    cout << "post: " << post_transform.matrix() << endl;

  }

  // Publish what we've been requested to publish
  for (auto it=publish_infos_.begin(); it != publish_infos_.end(); it++){
    // find this robot in the robot names
    for (int roboti=1; roboti < robot_names_.size(); roboti++){
      if (robot_names_[roboti] == it->robot_name){

        // publish state?
        if (it->publish_type == "state"){
          bot_core::robot_state_t manipulation_state;
          manipulation_state.utime = getUnixTime();
          std::string robot_name = robot_names_[roboti];

          manipulation_state.num_joints = 0;
          bool found_floating = false;
          for (int i=0; i<robot_->bodies.size(); i++){
            if (robot_->bodies[i]->get_model_name() == robot_name){
              if (robot_->bodies[i]->getJoint().is_floating()){
                Vector3d xyz = post_transform*x_.block<3, 1>(robot_->bodies[i]->get_position_start_index() + 0, 0);
                manipulation_state.pose.translation.x = xyz[0];
                manipulation_state.pose.translation.y = xyz[1];
                manipulation_state.pose.translation.z = xyz[2];
                Quaterniond quat1(post_transform.rotation());
                auto quat2 = rpy2quat(x_.block<3, 1>(robot_->bodies[i]->get_position_start_index() + 3, 0));
                quat1 *= Quaterniond(quat2[0], quat2[1], quat2[2], quat2[3]);
                manipulation_state.pose.rotation.w = quat1.w();
                manipulation_state.pose.rotation.x = quat1.x();
                manipulation_state.pose.rotation.y = quat1.y();
                manipulation_state.pose.rotation.z = quat1.z();
                if (found_floating){
                  printf("Had more than one floating joint???\n");
                  exit(-1);
                }
                found_floating = true;
              } else {
                // warning: if numpositions != numvelocities, problems arise...
                manipulation_state.num_joints += robot_->bodies[i]->getJoint().get_num_positions();
                for (int j=0; j < robot_->bodies[i]->getJoint().get_num_positions(); j++){
                  manipulation_state.joint_name.push_back(robot_->bodies[i]->getJoint().get_position_name(j));
                  manipulation_state.joint_position.push_back(x_[robot_->bodies[i]->get_position_start_index() + j]);
                  manipulation_state.joint_velocity.push_back(x_[robot_->bodies[i]->get_position_start_index() + j + robot_->get_num_positions()]);
                }
              }
            }
          }
          manipulation_state.joint_effort.resize(manipulation_state.num_joints, 0.0);
          std::string channelname = it->publish_channel;
          lcm_->publish(channelname, &manipulation_state);
        }

        // publish just the floating base transform?
        else if (it->publish_type == "transform") {
          bot_core::rigid_transform_t floating_base_transform;
          floating_base_transform.utime = getUnixTime();

          bool found_floating = false;
          for (int i=0; i<robot_->bodies.size(); i++){
            if (robot_->bodies[i]->get_model_name() == robot_names_[roboti]){
              if (robot_->bodies[i]->getJoint().is_floating()){
                Vector3d xyz = post_transform*x_.block<3, 1>(robot_->bodies[i]->get_position_start_index() + 0, 0);
                floating_base_transform.trans[0] = xyz[0];
                floating_base_transform.trans[1] = xyz[1];
                floating_base_transform.trans[2] = xyz[2];
                Quaterniond quat1(post_transform.rotation());
                auto quat2 = rpy2quat(x_.block<3, 1>(robot_->bodies[i]->get_position_start_index() + 3, 0));
                quat1 *= Quaterniond(quat2[0], quat2[1], quat2[2], quat2[3]);
                floating_base_transform.quat[0] = quat1.w();
                floating_base_transform.quat[1] = quat1.x();
                floating_base_transform.quat[2] = quat1.y();
                floating_base_transform.quat[3] = quat1.z();
                if (found_floating){
                  printf("Had more than one floating joint???\n");
                  exit(-1);
                }
                found_floating = true;
              }
            }
          }
          std::string channelname = it->publish_channel;
          lcm_->publish(channelname, &floating_base_transform);
        }

      }
    }
  }

  // Publish the object state
  //cout << "robot robot name vector: " << robot->robot_name.size() << endl;
  for (int roboti=1; roboti < robot_names_.size(); roboti++){
    

  }
}