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
  if (config["world_frame"])
    world_frame_ = config["world_frame"].as<bool>();

  if (verbose_lcmgl)
    lcmgl_tag_ = bot_lcmgl_init(lcm->getUnderlyingLCM(), (std::string("at_optotrak_") + robot_name).c_str());


  if (config["markers"]){
    int numtags = 0;
    for (auto iter=config["markers"].begin(); iter!=config["markers"].end(); iter++){
      MarkerAttachment attachment;

      attachment.list_id = numtags;
      numtags++;

      // first try to find the robot name
      std:string linkname = (*iter)["body"].as<string>();
      auto search = robot->findLink(linkname, robot_name);
      if (search == nullptr){
        printf("Couldn't find link name %s on robot %s", linkname.c_str(), robot_name.c_str());
        exit(1);
      }
      attachment.body_id = search->body_index;
      
      // and parse transformation
      
      Vector3d trans(Vector3d((*iter)["pos"][0].as<double>(), (*iter)["pos"][1].as<double>(), (*iter)["pos"][2].as<double>()));
      attachment.body_transform.setIdentity();
      attachment.body_transform.matrix().block<3, 1>(0,3) = trans;

      attachment.last_received = getUnixTime() - timeout_time*2.;

      int id = (*iter)["id"].as<double>();
      attachedMarkers[id] = attachment;
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
  for (auto it = attachedMarkers.begin(); it != attachedMarkers.end(); it++){
    MarkerAttachment * attachment = &(it->second);
    // actual transform
    Transform<double, 3, Isometry> current_transform =  robot->relativeTransform(robot_kinematics_cache, 0,  attachment->body_id);

    // spawn transform from state variables
    Transform<double, 3, Isometry> body_transform = attachment->body_transform;
    
    // actual transform xyz jacobian
    auto J_xyz = robot->transformPointsJacobian(robot_kinematics_cache, Vector3d(0.0, 0.0, 0.0), attachment->body_id, 0, false);
    //J_rpy.block<3, 3>(0, 3) = attachment->body_transform.rotation() * J_rpy.block<3, 3>(0, 3);

    // weird transforms to be thinking about everything in body from of this link
    Vector3d z_current = current_transform * Vector3d(0.0, 0.0, 0.0);
    Vector3d z_des = attachment->last_transform * body_transform.inverse() * Vector3d(0.0, 0.0, 0.0);
   // Vector4d quat_current = rotmat2quat(current_transform.rotation());
   // Vector4d quat_des = rotmat2quat(attachment->last_transform.rotation());


    // corners of object
    Matrix3Xd points(3, 8); 
    double width = 0.0763;
    points << Vector3d(-width/2., -width/2., 0.0),
          Vector3d(width/2., -width/2., 0.0),
          Vector3d(width/2., width/2., 0.0),
          Vector3d(-width/2., width/2., 0.0),
          Vector3d(0.0, 0.0, 0.0),
          Vector3d(0.1, 0.0, 0.0),
          Vector3d(0.0, 0.1, 0.0),
          Vector3d(0.0, 0.0, 0.1);
    Matrix3Xd points_cur =  current_transform * body_transform * points;
    Matrix3Xd points_des =  attachment->last_transform * points;

    Vector3d body_trans_offset = body_transform * Vector3d(0.0, 0.0, 0.0);

    if (verbose_lcmgl)
    {
      bot_lcmgl_begin(lcmgl_tag_, LCMGL_QUADS);
      bot_lcmgl_color3f(lcmgl_tag_, fmin(1.0, fabs(body_trans_offset(0)/0.05)),
                                    fmin(1.0, fabs(body_trans_offset(1)/0.05)),
                                    fmin(1.0, fabs(body_trans_offset(2)/0.05)));
      bot_lcmgl_line_width(lcmgl_tag_, 4.0f);
      for (int i=0; i < 4; i++)
        bot_lcmgl_vertex3f(lcmgl_tag_, points_cur(0, i), points_cur(1, i), points_cur(2, i));

      bot_lcmgl_end(lcmgl_tag_); 

      bot_lcmgl_begin(lcmgl_tag_, LCMGL_LINES);
      bot_lcmgl_line_width(lcmgl_tag_, 4.0f);
      bot_lcmgl_color3f(lcmgl_tag_, 1.0, 0.5, 0.0);
      bot_lcmgl_vertex3f(lcmgl_tag_, points_cur(0, 4), points_cur(1, 4), points_cur(2, 4));
      bot_lcmgl_vertex3f(lcmgl_tag_, points_cur(0, 5), points_cur(1, 5), points_cur(2, 5));
      bot_lcmgl_color3f(lcmgl_tag_, 0.0, 1.0, 0.0);
      bot_lcmgl_vertex3f(lcmgl_tag_, points_cur(0, 4), points_cur(1, 4), points_cur(2, 4));
      bot_lcmgl_vertex3f(lcmgl_tag_, points_cur(0, 6), points_cur(1, 6), points_cur(2, 6));
      bot_lcmgl_color3f(lcmgl_tag_, 0.0, 0.5, 1.0);
      bot_lcmgl_vertex3f(lcmgl_tag_, points_cur(0, 4), points_cur(1, 4), points_cur(2, 4));
      bot_lcmgl_vertex3f(lcmgl_tag_, points_cur(0, 7), points_cur(1, 7), points_cur(2, 7));
      bot_lcmgl_end(lcmgl_tag_);
    }


    if (now - attachment->last_received < timeout_time){
      VectorXd z_c(3);
      z_c << z_current;
      VectorXd z_d(3);
      z_d << z_des;
      MatrixXd J(3, robot->number_of_positions());
      J << J_xyz;

      // POSITION FROM DETECTED TRANSFORM:
      // 0.5 * x.' Q x + f.' x
      // position error:
      // min (z_current - z_des)^2
      // min ( (z_current + J*(q_new - q_old)) - z_des )^2
      MatrixXd Ks = z_c - z_d - J*q_old;
      f.block(0, 0, nq, 1) += MARKER_WEIGHT * (2. * Ks.transpose() * J).transpose();
      Q.block(0, 0, nq, nq) += MARKER_WEIGHT * (2. * J.transpose() * J);
      K += MARKER_WEIGHT * Ks.squaredNorm();

      if (verbose_lcmgl){
        bot_lcmgl_begin(lcmgl_tag_, LCMGL_QUADS);
        bot_lcmgl_color3f(lcmgl_tag_, 0.5, 0.5, 0.5);
        bot_lcmgl_line_width(lcmgl_tag_, 4.0f);
        for (int i=0; i < 4; i++)
          bot_lcmgl_vertex3f(lcmgl_tag_, points_des(0, i), points_des(1, i), points_des(2, i));
        bot_lcmgl_end(lcmgl_tag_); 

        bot_lcmgl_begin(lcmgl_tag_, LCMGL_LINES);
        bot_lcmgl_line_width(lcmgl_tag_, 4.0f);
        bot_lcmgl_color3f(lcmgl_tag_, 1.0, 0.0, 0.5);
        bot_lcmgl_vertex3f(lcmgl_tag_, points_des(0, 4), points_des(1, 4), points_des(2, 4));
        bot_lcmgl_vertex3f(lcmgl_tag_, points_des(0, 5), points_des(1, 5), points_des(2, 5));
        bot_lcmgl_color3f(lcmgl_tag_, 0.0, 1.0, 0.5);
        bot_lcmgl_vertex3f(lcmgl_tag_, points_des(0, 4), points_des(1, 4), points_des(2, 4));
        bot_lcmgl_vertex3f(lcmgl_tag_, points_des(0, 6), points_des(1, 6), points_des(2, 6));
        bot_lcmgl_color3f(lcmgl_tag_, 0.0, 0.0, 1.0);
        bot_lcmgl_vertex3f(lcmgl_tag_, points_des(0, 4), points_des(1, 4), points_des(2, 4));
        bot_lcmgl_vertex3f(lcmgl_tag_, points_des(0, 7), points_des(1, 7), points_des(2, 7));
        bot_lcmgl_end(lcmgl_tag_);
      }

      if (verbose){
        cout << endl << endl << endl << "********* MARKER " << it->first << " **************" << endl;
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

  for (int i=0; i < msg->number_rigid_bodies; i++){
    auto it = attachedMarkers.find(i+1);
    if (it != attachedMarkers.end()){
      if (msg->z[i] <= -500.0 && msg->z[i] >= -20000.){ // marker is in front of camera but in range
        printf("populated marker %d\n", i);
        (it->second).last_received = getUnixTime();
        (it->second).last_transform.setIdentity();
        (it->second).last_transform.matrix().block<3, 1>(0,3) = Vector3d(msg->x[i]/1000., msg->y[i]/1000., msg->z[i]/1000.);
      }
    }
  }

  detectionsMutex.unlock();
}