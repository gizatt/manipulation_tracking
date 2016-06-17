#undef NDEBUG
#include <assert.h> 
#include <fstream>
#include "OptotrakMarkerCost.hpp"
#include "drake/util/convexHull.h"
#include "zlib.h"
#include <cmath>
#include "common.hpp"

using namespace std;
using namespace Eigen;

OptotrakMarkerCost::OptotrakMarkerCost(std::shared_ptr<const RigidBodyTree> robot_, std::shared_ptr<lcm::LCM> lcm_, YAML::Node config) :
    robot(robot_),
    robot_kinematics_cache(robot->bodies),
    lcm(lcm_),
    nq(robot->number_of_positions())
{
  if (config["attached_manipuland"]){
    // try to find this robot
    robot_name = config["attached_manipuland"].as<string>();
  }

  const char * filename = NULL;
  if (config["filename"])
    filename = config["filename"].as<string>().c_str();
  this->initBotConfig(filename);

  if (config["localization_var"])
    localization_var = config["localization_var"].as<double>();
  if (config["transform_var"])
    transform_var = config["transform_var"].as<double>();
  if (config["timeout_time"])
    timeout_time = config["timeout_time"].as<double>();
  if (config["verbose"])
    verbose = config["verbose"].as<bool>();
  if (config["verbose_lcmgl"])
    verbose_lcmgl = config["verbose_lcmgl"].as<bool>();
  if (config["free_floating_base"])
    free_floating_base_ = config["free_floating_base"].as<bool>();

  if (verbose_lcmgl)
    lcmgl_tag_ = bot_lcmgl_init(lcm->getUnderlyingLCM(), (std::string("at_optotrak_") + robot_name).c_str());


  if (config["markers"]){
    for (auto iter=config["markers"].begin(); iter!=config["markers"].end(); iter++){
      MarkerAttachment attachment;

      // first try to find the robot name
      string linkname = (*iter)["body"].as<string>();
      auto search = robot->findLink(linkname, robot_name);
      if (search == nullptr){
        printf("Couldn't find link name %s on robot %s", linkname.c_str(), robot_name.c_str());
        exit(1);
      }
      attachment.body_id = search->body_index;
      attachment.marker_ids = (*iter)["ids"].as<vector<int>>();
      // shift from 1-index to 0-index
      for (int i=0; i < attachment.marker_ids.size(); i++) attachment.marker_ids[i] -= 1;

      // and parse transformation
      Vector3d trans(Vector3d((*iter)["pos"][0].as<double>(), (*iter)["pos"][1].as<double>(), (*iter)["pos"][2].as<double>()));
      attachment.body_transform.setIdentity();
      attachment.body_transform.matrix().block<3, 1>(0,3) = trans;

      attachment.last_received = getUnixTime() - timeout_time*2.;

      attachedMarkers.push_back(attachment);
    }
  }


  auto marker_sub = lcm->subscribe("irb140_gelsight_markers", &OptotrakMarkerCost::handleOptotrakMarkerMessage, this);
  marker_sub->setQueueCapacity(1);

}


void OptotrakMarkerCost::initBotConfig(const char* filename)
{
  if (filename && filename[0])
    {
      botparam_ = bot_param_new_from_file(filename);
    }
  else
    {
    while (!botparam_)
      {
        botparam_ = bot_param_new_from_server(lcm->getUnderlyingLCM(), 0);
      }
    }
  botframes_ = bot_frames_get_global(lcm->getUnderlyingLCM(), botparam_);
}

int OptotrakMarkerCost::get_trans_with_utime(std::string from_frame, std::string to_frame,
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


bool OptotrakMarkerCost::constructCost(ManipulationTracker * tracker, const Eigen::Matrix<double, Eigen::Dynamic, 1> x_old, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Q, Eigen::Matrix<double, Eigen::Dynamic, 1>& f, double& K)
{
  double now = getUnixTime();

  double MARKER_WEIGHT = std::isinf(localization_var) ? 0.0 : 1. / (2. * localization_var * localization_var);
  double BODY_TRANSFORM_WEIGHT = std::isinf(transform_var) ? 0.0 : 1. / (2. * transform_var * transform_var);

  VectorXd q_old = x_old.block(0, 0, robot->number_of_positions(), 1);
  robot_kinematics_cache.initialize(q_old);
  robot->doKinematics(robot_kinematics_cache);

  detectionsMutex.lock();
  for (auto attachment = attachedMarkers.begin(); attachment != attachedMarkers.end(); attachment++){
    Transform<double, 3, Isometry> body_transform = attachment->body_transform;

    // actual transform
    Transform<double, 3, Isometry> current_transform = robot->relativeTransform(robot_kinematics_cache, 0, attachment->body_id);

    
    // actual transform xyz jacobian
    auto J_xyz = robot->transformPointsJacobian(robot_kinematics_cache, body_transform * Vector3d(0.0, 0.0, 0.0), attachment->body_id, 0, false);
   //J_rpy.block<3, 3>(0, 3) = attachment->body_transform.rotation() * J_rpy.block<3, 3>(0, 3);

    Vector3d z_current = current_transform * body_transform * Vector3d(0.0, 0.0, 0.0);
    Vector3d z_des = attachment->last_transform * Vector3d(0.0, 0.0, 0.0);
   // Vector4d quat_current = rotmat2quat(current_transform.rotation());
   // Vector4d quat_des = rotmat2quat(attachment->last_transform.rotation());


    Vector3d body_trans_offset = body_transform * Vector3d(0.0, 0.0, 0.0);

    
    if (verbose_lcmgl)
    {
      bot_lcmgl_begin(lcmgl_tag_, LCMGL_POINTS);
      bot_lcmgl_color3f(lcmgl_tag_, fmin(1.0, fabs(body_trans_offset(0)/0.05)),
                                    fmin(1.0, fabs(body_trans_offset(1)/0.05)),
                                    fmin(1.0, fabs(body_trans_offset(2)/0.05)));
      bot_lcmgl_vertex3f(lcmgl_tag_, z_current[0], z_current[1], z_current[2]);
      bot_lcmgl_color3f(lcmgl_tag_, 1.0, 0.5, 0.0);
      bot_lcmgl_vertex3f(lcmgl_tag_, z_des[0], z_des[1], z_des[2]);
      bot_lcmgl_end(lcmgl_tag_);
    }


    if (now - attachment->last_received < timeout_time){
      VectorXd z_c(3);
      z_c << z_current;
      VectorXd z_d(3);
      z_d << z_des;
      MatrixXd J(3, robot->number_of_positions());
      J << J_xyz; //, J_rpy;

      // POSITION FROM DETECTED TRANSFORM:
      // 0.5 * x.' Q x + f.' x
      // position error:
      // min (z_current - z_des)^2
      // min ( (z_current + J*(q_new - q_old)) - z_des )^2
      MatrixXd Ks = z_c - z_d - J*q_old;

      if (free_floating_base_){
        f.block(6, 0, nq-6, 1) += MARKER_WEIGHT * (2. * Ks.transpose() * J).transpose().block(6, 0, nq-6, 1);
        Q.block(6, 6, nq-6, nq-6) += MARKER_WEIGHT * (2. * J.transpose() * J).block(6, 6, nq-6, nq-6);
      } else {
        f.block(0, 0, nq, 1) += MARKER_WEIGHT * (2. * Ks.transpose() * J).transpose();
        Q.block(0, 0, nq, nq) += MARKER_WEIGHT * (2. * J.transpose() * J);
      }
      K += MARKER_WEIGHT * Ks.squaredNorm();

      if (verbose){
        cout << endl << endl << endl << "****** MARKERS [";
        for (auto it=attachment->marker_ids.begin(); it != attachment->marker_ids.end(); it++) cout << *it << ", ";
        cout << "] ***********" << endl;
        cout << "Body transform: " << body_transform.matrix() << endl;
        cout << "J: " << J << endl;
        cout << "z_c: " << z_c.transpose() << endl;
        cout << "z_d: " << z_d.transpose() << endl;
        cout << "Q old: " << q_old.transpose() << endl;
        cout << "KS: " << Ks.transpose() << endl;
      }
    }
  }
  detectionsMutex.unlock();

  if (verbose_lcmgl)
    bot_lcmgl_switch_buffer(lcmgl_tag_);  

  if (verbose)
    printf("Spent %f in optotrak constraints.\n", getUnixTime() - now);
  return true;
}

void OptotrakMarkerCost::handleOptotrakMarkerMessage(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const drake::lcmt_optotrak * msg){
  printf("Received marker list on channel  %s\n", chan.c_str());

  detectionsMutex.lock();

  for (auto it = attachedMarkers.begin(); it != attachedMarkers.end(); it++){
    bool all_present = true;
    double avg_x = 0.0;
    double avg_y = 0.0;
    double avg_z = 0.0;
    for (auto marker_id = it->marker_ids.begin(); marker_id != it->marker_ids.end(); marker_id++){
      if (*marker_id < 0 || *marker_id >= msg->number_rigid_bodies || msg->z[*marker_id] <= -20000 ||
          msg->z[*marker_id] >= -500){
        all_present = false;
        break;
      } else {
        avg_x += msg->x[*marker_id];
        avg_y += msg->y[*marker_id];
        avg_z += msg->z[*marker_id];
      }
    }

    if (all_present){
      avg_x /= (double)it->marker_ids.size();
      avg_y /= (double)it->marker_ids.size();
      avg_z /= (double)it->marker_ids.size();
      it->last_received = getUnixTime();
      it->last_transform.setIdentity();
      it->last_transform.matrix().block<3, 1>(0,3) = Vector3d(avg_x/1000., avg_y/1000., avg_z/1000.);
    }
  }



  detectionsMutex.unlock();
}