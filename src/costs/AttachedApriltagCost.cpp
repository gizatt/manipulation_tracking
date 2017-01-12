#undef NDEBUG
#include <assert.h> 
#include <fstream>
#include "AttachedApriltagCost.hpp"
#include "drake/util/convexHull.h"
#include "drake/util/drakeGeometryUtil.h"
#include <cmath>
#include "common/common.hpp"

using namespace std;
using namespace Eigen;
using namespace drake::math;

AttachedApriltagCost::AttachedApriltagCost(std::shared_ptr<const RigidBodyTree<double> > robot_, std::shared_ptr<lcm::LCM> lcm_, YAML::Node config) :
    robot(robot_),
    lcm(lcm_),
    nq(robot->get_num_positions())
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
    lcmgl_tag_ = bot_lcmgl_init(lcm->getUnderlyingLCM(), (std::string("at_apr_") + robot_name).c_str());


  if (config["apriltags"]){
    int numtags = 0;
    for (auto iter=config["apriltags"].begin(); iter!=config["apriltags"].end(); iter++){
      ApriltagAttachment attachment;

      attachment.list_id = numtags;
      numtags++;

      // first try to find the robot name
      std:string linkname = (*iter)["body"].as<string>();
      auto search = robot->FindBody(linkname, robot_name);
      if (search == nullptr){
        printf("Couldn't find link name %s on robot %s", linkname.c_str(), robot_name.c_str());
        exit(1);
      }
      attachment.body_id = search->get_body_index();
      
      // and parse transformation
      
      Vector3d trans(Vector3d((*iter)["pos"][0].as<double>(), (*iter)["pos"][1].as<double>(), (*iter)["pos"][2].as<double>()));
      Vector3d rpy((*iter)["rot"][0].as<double>()*M_PI/180., (*iter)["rot"][1].as<double>()*M_PI/180., (*iter)["rot"][2].as<double>()*M_PI/180.);
      Quaterniond rot = AngleAxisd(rpy[2], Vector3d::UnitZ())
                  *  AngleAxisd(rpy[1], Vector3d::UnitY())
                  *  AngleAxisd(rpy[0], Vector3d::UnitX());
      attachment.body_transform.setIdentity();
      attachment.body_transform.matrix().block<3, 3>(0,0) = rot.matrix();
      attachment.body_transform.matrix().block<3, 1>(0,3) = trans;

      // create a 6 state variable transform initiatized with this transform
      num_extra_vars_ += 6;
      extra_vars_x0_.conservativeResize(num_extra_vars_, 1);
      extra_vars_x0_.block(num_extra_vars_ - 6, 0, 3, 1) = trans;
      extra_vars_x0_.block(num_extra_vars_ - 3, 0, 3, 1) = rpy;

      attachment.last_received = getUnixTime() - timeout_time*2.;
      std::string channel = (*iter)["channel"].as<string>();
      attachment.detection_sub = lcm->subscribe(channel, &AttachedApriltagCost::handleTagDetectionMsg, this);
      attachment.detection_sub->setQueueCapacity(1);

      int id = (*iter)["id"].as<double>();
      attachedApriltags[id] = attachment;

      channelToApriltagIndex[channel] = id;
    }
  }


  auto camera_offset_sub = lcm->subscribe("GT_CAMERA_OFFSET", &AttachedApriltagCost::handleCameraOffsetMsg, this);
  camera_offset_sub->setQueueCapacity(1);

}


void AttachedApriltagCost::initBotConfig(const char* filename)
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

int AttachedApriltagCost::get_trans_with_utime(std::string from_frame, std::string to_frame,
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


bool AttachedApriltagCost::constructCost(ManipulationTracker * tracker, const Eigen::Matrix<double, Eigen::Dynamic, 1> x_old, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Q, Eigen::Matrix<double, Eigen::Dynamic, 1>& f, double& K)
{
  double now = getUnixTime();

  double ATTACHED_APRILTAG_WEIGHT = std::isinf(localization_var) ? 0.0 : 1. / (2. * localization_var * localization_var);
  double BODY_TRANSFORM_WEIGHT = std::isinf(transform_var) ? 0.0 : 1. / (2. * transform_var * transform_var);

  VectorXd q_old = x_old.block(0, 0, robot->get_num_positions(), 1);
  auto robot_kinematics_cache = robot->doKinematics(q_old);

  // get transform from camera to world frame
  Eigen::Isometry3d kinect2world;
  if (world_frame_){
    long long utime = 0;
    Eigen::Isometry3d robot2world;
    this->get_trans_with_utime("robot_base", "local", utime, robot2world);
    long long utime2 = 0;
    camera_offset_mutex.lock();
    kinect2world =  robot2world * kinect2robot.inverse();
    camera_offset_mutex.unlock();
  } else {
    kinect2world.setIdentity();
  }

  detectionsMutex.lock();
  for (auto it = attachedApriltags.begin(); it != attachedApriltags.end(); it++){
    ApriltagAttachment * attachment = &(it->second);
    // actual transform
    Transform<double, 3, Isometry> current_transform =  robot->relativeTransform(robot_kinematics_cache, 0,  attachment->body_id);

    // spawn transform from state variables
    int start_ind = robot->get_num_positions() + robot->get_num_velocities() + 6*attachment->list_id;
    Vector3d trans(x_old.block(start_ind, 0, 3, 1));
    Vector3d rpy(x_old.block(start_ind + 3, 0, 3, 1));
    Quaterniond rot = AngleAxisd(rpy[2], Vector3d::UnitZ())
                  *  AngleAxisd(rpy[1], Vector3d::UnitY())
                  *  AngleAxisd(rpy[0], Vector3d::UnitX());
    Transform<double, 3, Isometry> body_transform;
    body_transform.setIdentity();
    body_transform.matrix().block<3, 3>(0,0) = rot.matrix();
    body_transform.matrix().block<3, 1>(0,3) = trans;
    
    // actual transform xyz jacobian
    auto J_xyz = robot->transformPointsJacobian(robot_kinematics_cache, Vector3d(0.0, 0.0, 0.0), attachment->body_id, 0, false);
    auto J_quat = robot->relativeQuaternionJacobian(robot_kinematics_cache, attachment->body_id, 0, false);
    auto J_rpy = robot->relativeRollPitchYawJacobian(robot_kinematics_cache, attachment->body_id, 0, false);
    //J_rpy.block<3, 3>(0, 3) = attachment->body_transform.rotation() * J_rpy.block<3, 3>(0, 3);

    // weird transforms to be thinking about everything in body from of this link
    Vector3d z_current = current_transform * Vector3d(0.0, 0.0, 0.0);
    Vector3d z_des = kinect2world * attachment->last_transform * body_transform.inverse() * Vector3d(0.0, 0.0, 0.0);
    Vector3d rpy_current = rotmat2rpy(current_transform.rotation());
    Vector3d rpy_des =  rotmat2rpy((kinect2world * attachment->last_transform * body_transform.inverse()).rotation());
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
    Matrix3Xd points_des =  kinect2world * attachment->last_transform * points;

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
      VectorXd z_c(6);
      z_c << z_current, rpy_current;
      VectorXd z_d(6);
      z_d << z_des, rpy_des;
      MatrixXd J(6, robot->get_num_positions());
      J << J_xyz, J_rpy;

      // POSITION FROM DETECTED TRANSFORM:
      // 0.5 * x.' Q x + f.' x
      // position error:
      // min (z_current - z_des)^2
      // min ( (z_current + J*(q_new - q_old)) - z_des )^2
      MatrixXd Ks = z_c - z_d - J*q_old;
      f.block(0, 0, nq, 1) += ATTACHED_APRILTAG_WEIGHT * (2. * Ks.transpose() * J).transpose();
      Q.block(0, 0, nq, nq) += ATTACHED_APRILTAG_WEIGHT * (2. * J.transpose() * J);
      K += ATTACHED_APRILTAG_WEIGHT * Ks.squaredNorm();

      // BODY LINK TRANSFORM 
      // penalize from predicted body transform given the global transform and the object pose
      Transform<double, 3, Isometry> error_transform = current_transform.inverse() * attachment->last_transform;
      VectorXd error_trans_exp(6);
      error_trans_exp.setZero();

      error_trans_exp.block<3, 1>(0, 0) = error_transform.matrix().block<3, 1>(0,3);
      error_trans_exp.block<3, 1>(3, 0) = rpy; //(rpy_des - rpy_current); //error_transform.rotation().eulerAngles(0, 1, 2);

      f.block(start_ind, 0, 6, 1) -= BODY_TRANSFORM_WEIGHT * error_trans_exp;
      Q.block(start_ind, start_ind, 6, 6) += BODY_TRANSFORM_WEIGHT * MatrixXd::Identity(6, 6);
      K += BODY_TRANSFORM_WEIGHT * error_trans_exp.transpose() * error_trans_exp;


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
        cout << endl << endl << endl << "********* TAG " << it->first << " **************" << endl;
        cout << "Body trans: " << endl << body_transform.matrix() << endl;
        cout << "J: " << J << endl;
        cout << "z_c: " << z_c.transpose() << endl;
        cout << "z_d: " << z_d.transpose() << endl;
        cout << "Q old: " << q_old.transpose() << endl;
        cout << "KS: " << Ks.transpose() << endl;
      }

      /*
      // position error:
      // min (z_current - z_des)^2
      // min ( (z_current + J*(q_new - q_old)) - z_des )^2
      MatrixXd Ks = z_current - z_des + J_xyz*q_old;
      cout << "Curr: " << current_transform.matrix() << endl;
      cout << "Last: " << attachment->last_transform.matrix() << endl;
      cout << "Q old: " << q_old << endl;
      cout << "KS: " << Ks << endl;
      cout << "JXYZ: " << J_xyz << endl;
      f -= ATTACHED_APRILTAG_WEIGHT * (2. * Ks.transpose() * J_xyz).transpose();
      Q += ATTACHED_APRILTAG_WEIGHT * (2. * J_xyz.transpose() * J_xyz);
      K += ATTACHED_APRILTAG_WEIGHT * Ks.squaredNorm();
*/

      // rotation error, via quaternion distance error metric from 
      // https://www-preview.ri.cmu.edu/pub_files/pub4/kuffner_james_2004_1/kuffner_james_2004_1.pdf
      /*
      double dot = quat_current.transpose() * quat_des;*/
      // min 1 - (quat_des.' * quat_curr)^2
      // min 1 - (quat_des.' * (quat_curr + J_quat * qdiff))^2
      // expand waaaay out
      /*
      K += ATTACHED_APRILTAG_WEIGHT * (1. - quat_current.transpose() * quat_des * quat_des.transpose() * quat_current 
        - 2 * quat_des * quat_des.transpose() * quat_current.transpose() * J_quat.transpose() * q_old
        - q_old.transpose() * J_quat * quat_des * quat_des.transpose() * J_quat.transpose() * q_old);
      f += ATTACHED_APRILTAG_WEIGHT * (-2.*quat_des*quat_des.transpose()*quat_current.transpose()*J_quat.transpose()
        + 2 * q_old * J_quat * quat_des * quat_des.transpose() * J_quat.transpose());
        
      Q += ATTACHED_APRILTAG_WEIGHT * (-1. * J_quat * quat_des * quat_des.transpose() * J_quat.transpose());
      */
    }
  }
  detectionsMutex.unlock();

  if (verbose_lcmgl)
    bot_lcmgl_switch_buffer(lcmgl_tag_);  

  if (verbose)
    printf("Spent %f in attached apriltag constraints.\n", getUnixTime() - now);
  return true;
}

void AttachedApriltagCost::handleTagDetectionMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::rigid_transform_t* msg){
  //printf("Received hand state on channel  %s\n", chan.c_str());

  auto it = channelToApriltagIndex.find(chan);
  if (it == channelToApriltagIndex.end()){
    printf("Received message from channel we didn't subscribe to... panicking now.\n");
    exit(-1);
  }

  detectionsMutex.lock();

  ApriltagAttachment * attachment = &attachedApriltags[it->second];
  attachment->last_received = getUnixTime();

  Quaterniond rot(msg->quat[0], msg->quat[1], msg->quat[2], msg->quat[3]);
  attachment->last_transform.setIdentity();
  attachment->last_transform.matrix().block<3, 3>(0,0) = rot.matrix();
  attachment->last_transform.matrix().block<3, 1>(0,3) = Vector3d(msg->trans[0], msg->trans[1], msg->trans[2]);

  detectionsMutex.unlock();
}


void AttachedApriltagCost::handleCameraOffsetMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::rigid_transform_t* msg){
  camera_offset_mutex.lock();
  Vector3d trans(msg->trans[0], msg->trans[1], msg->trans[2]);
  Quaterniond rot(msg->quat[0], msg->quat[1], msg->quat[2], msg->quat[3]);
  kinect2robot.setIdentity();
  kinect2robot.matrix().block<3, 3>(0,0) = rot.matrix();
  kinect2robot.matrix().block<3, 1>(0,3) = trans;
  camera_offset_mutex.unlock();
}
