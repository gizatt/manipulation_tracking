#undef NDEBUG
#include <assert.h> 
#include <fstream>
#include "OptotrakMarkerCost.hpp"
#include "drake/util/convexHull.h"
#include "drake/util/drakeGeometryUtil.h"
#include <cmath>
#include "common/common.hpp"

using namespace std;
using namespace Eigen;
using namespace drake::math;

Matrix3d calcS(double x, double y, double z){
  Matrix3d S;
  S << 0, -z, y,
          z, 0, -x,
          -y, x, 0;
  return S;
}

OptotrakMarkerCost::OptotrakMarkerCost(std::shared_ptr<const RigidBodyTree<double> > robot_, std::shared_ptr<lcm::LCM> lcm_, YAML::Node config) :
    robot(robot_),
    robot_kinematics_cache(robot->get_num_positions(), robot->get_num_velocities()),
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
  if (config["free_floating_base"])
    free_floating_base_ = config["free_floating_base"].as<bool>();

  if (verbose_lcmgl)
    lcmgl_tag_ = bot_lcmgl_init(lcm->getUnderlyingLCM(), (std::string("at_optotrak_") + robot_name).c_str());


  if (config["markers"]){
    for (auto iter=config["markers"].begin(); iter!=config["markers"].end(); iter++){
      MarkerAttachment attachment;

      // first try to find the robot name
      string linkname = (*iter)["body"].as<string>();
      auto search = robot->FindBody(linkname, robot_name);
      if (search == nullptr){
        printf("Couldn't find link name %s on robot %s", linkname.c_str(), robot_name.c_str());
        exit(1);
      }
      attachment.body_id = search->get_body_index();
      attachment.marker_ids = (*iter)["ids"].as<vector<int>>();

      // shift from 1-index to 0-index
      for (int i=0; i < attachment.marker_ids.size(); i++) attachment.marker_ids[i] -= 1;

      // pull normal if one is supplied and we can act on it
      if (attachment.marker_ids.size() >= 3 && (*iter)["normal"]){
        attachment.normal = Vector3d((*iter)["normal"][0].as<double>(),
                                     (*iter)["normal"][1].as<double>(),
                                     (*iter)["normal"][2].as<double>());
        // normalize just to be sure
        attachment.normal /= attachment.normal.norm();
        attachment.have_normal = true;
      } else {
        attachment.normal.setZero();
        attachment.have_normal = false;
      }

      attachment.have_last_normal = false;

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

  VectorXd q_old = x_old.block(0, 0, robot->get_num_positions(), 1);
  robot_kinematics_cache.initialize(q_old);
  robot->doKinematics(robot_kinematics_cache);

  detectionsMutex.lock();
  for (auto attachment = attachedMarkers.begin(); attachment != attachedMarkers.end(); attachment++){
    Transform<double, 3, Isometry> body_transform = attachment->body_transform;

    // actual transform
    Transform<double, 3, Isometry> current_transform = robot->relativeTransform(robot_kinematics_cache, 0, attachment->body_id);

    
    // actual transform xyz and rpy jacobian
    auto J_xyz = robot->transformPointsJacobian(robot_kinematics_cache, body_transform * Vector3d(0.0, 0.0, 0.0), attachment->body_id, 0, false);
    auto J_rpy = robot->relativeRollPitchYawJacobian(robot_kinematics_cache, attachment->body_id, 0, false);

    Vector3d z_current = current_transform * body_transform * Vector3d(0.0, 0.0, 0.0);
    Vector3d z_des = attachment->last_offset;
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
      MatrixXd J(3, robot->get_num_positions());
      J << J_xyz; //, J_rpy;

      // POSITION FROM DETECTED TRANSFORM:
      // 0.5 * x.' Q x + f.' x
      // Offset error:
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

      if (attachment->have_normal && attachment->have_last_normal){
        // We know the direction the cluster faces and the direction it should face
        // so enforce they point the same way

        // figure out transform error from http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
        Vector3d v = (current_transform.matrix().block<3, 3>(0, 0)*attachment->normal).cross(attachment->last_normal);
        double s = v.norm();
        double c = (current_transform.matrix().block<3, 3>(0, 0)*attachment->normal).transpose() * attachment->last_normal;
        Matrix3d correction_R = Matrix3d::Identity();
        Matrix3d S = calcS(v(0), v(1), v(2)) ;
        if (s > 0)
          correction_R += S + S*S*((1.-c)/(s*s));

        // get that into an rpy
        Vector3d correction_rpy = rotmat2rpy(correction_R);
        Vector3d curr_rpy = rotmat2rpy(current_transform.matrix().block<3, 3>(0, 0));
        // enforce that change in rpy via our jacobian
        // rpy = rpy_curr + J_rpy( q - q_old)
        // min (rpy - (rpy_curr + correction_rpy))^2
        // min (rpy_curr + J_rpy( q - q_old) - rpy_curr - correction_rpy))^2
        // which cancels and makes sense:
        // min (J_rpy( q - q_old) - correction_rpy)^2
        // min  q.' J_rpy^2 q  -2 (correction_rpy + 
        Q.block(0, 0, nq, nq) += BODY_TRANSFORM_WEIGHT * 2. * J_rpy.transpose() * J_rpy;
        f.block(0, 0, nq, 1) -= BODY_TRANSFORM_WEIGHT * 2. * ((correction_rpy + J_rpy*q_old).transpose()*J_rpy).transpose();
        K += BODY_TRANSFORM_WEIGHT * (correction_rpy + J_rpy*q_old).squaredNorm();

        if (verbose){
          cout << "******" << endl;
          cout << "Known normal, transformed: " << (current_transform.matrix().block<3, 3>(0, 0)*attachment->normal).transpose() << endl;
          cout << "New normal: " << attachment->last_normal.transpose() << endl;
          cout << "Curr rpy: " << curr_rpy.transpose() << endl;
          cout << "Correction rpy: " << correction_rpy.transpose() << endl;
          cout << "Jrpy: " << J_rpy << endl;
          cout << "******" << endl;
        }

        /*  
        // error = (1 - n_des.' * (R(q) * n_known)
        // linearizes to:
        // [1 - n_des.'(R(q_old) * n_known) + F.' * q_old] - F.' * q
        // for F_i = n_des.' * (dR/droll * droll/dq_i + ... + dR/dyaq * dyaq/dq_i) * n_known
        Matrix3d R_old = current_transform.matrix().block<3, 3>(0, 0);
        MatrixXd drotmat_drpy = drpy2rotmat(rotmat2rpy(R_old)); // 9 x 3
        Map<MatrixXd> drotmat_dr(drotmat_drpy.data(), 3,3);
        Map<MatrixXd> drotmat_dp(drotmat_drpy.data(), 3,3);
        Map<MatrixXd> drotmat_dy(drotmat_drpy.data(), 3,3);

        VectorXd F(nq);
        for (int i=0; i < nq; i++){
          // following http://arxiv.org/pdf/1311.6010.pdf
          // dR = S(Dtheta)*R
          F(i) = attachment->last_normal.transpose() * 
              (   
                (
                    drotmat_dr*J_rpy(0,i) +
                    drotmat_dp*J_rpy(1,i) +
                    drotmat_dy*J_rpy(2,i)
                ) * attachment->normal
              );
        }
        VectorXd C = VectorXd::Ones(1) - attachment->last_normal.transpose() * (R_old * attachment->normal) + F.transpose() * q_old;

        // Penalize quadratically: (C - F.' * q).' * (C - F.' * q)
        //                         C.' * C - 2 * C.' * F.' * q + q.' F * F.' * q
        if (free_floating_base_){
        } else {
          f.block(0, 0, nq, 1) -= BODY_TRANSFORM_WEIGHT * (2. * C.transpose() * F.transpose()).transpose();
          Q.block(0, 0, nq, nq) += BODY_TRANSFORM_WEIGHT * (2. * F * F.transpose());
        }
        K += BODY_TRANSFORM_WEIGHT * (C.squaredNorm());
        */


        // old bad
        // error = (1-normal_des.' * (R * normal))
        // linearized -> (err_curr) + J_err * (x - x old)
        // min norm -> ((err_curr - J_err*x_old) + J_err*x)^2
        //          -> (err_curr-J_err*x_old)^2 + 2*(err_curr-J_err*x_old)*J_err*x + x.' J_err.' J_err x; plus some transposes
        //MatrixXd err_curr = MatrixXd::Ones(1,1) - (attachment->last_normal.transpose() * current_transform.matrix().block<3, 3>(0, 0) * attachment->normal);
        /*
        VectorXd J_err(nq);

        for (int i=0; i < nq; i++){

          // following http://arxiv.org/pdf/1311.6010.pdf
          // dR = S(Dtheta)*R
          /*
          Vector3d derr_drpy;
          derr_drpy << -attachment->last_normal.transpose()* calcS(1, 0, 0)*current_transform.matrix().block<3, 3>(0, 0).transpose() * attachment->normal,
                       -attachment->last_normal.transpose()* calcS(0, 1, 0)*current_transform.matrix().block<3, 3>(0, 0).transpose() * attachment->normal,
                       -attachment->last_normal.transpose()* calcS(0, 0, 1)*current_transform.matrix().block<3, 3>(0, 0).transpose() * attachment->normal;
          J_err(i) = derr_drpy.transpose() * J_rpy.block<3,1>(0, i);
        }

        cout << endl << endl << endl;
        cout << "*************" << endl;
        cout << "Using normal " << (current_transform.matrix().block<3, 3>(0, 0) * attachment->normal).transpose() << " and current " << attachment->last_normal.transpose() << endl;
        cout << "Found err " << err_curr << " and J_err " << J_err.transpose() << endl;
        cout << "*************" << endl;

        if (free_floating_base_){
        } else {
          f.block(0, 0, nq, 1) += MARKER_WEIGHT/1000.0 * (2. * (err_curr - J_err.transpose()*q_old) * J_err.transpose()).transpose();
          Q.block(0, 0, nq, nq) += MARKER_WEIGHT/1000.0 * (2. * J_err * J_err.transpose()); // JErr already column vec
        }
        K += MARKER_WEIGHT/1000.0 * (err_curr - J_err.transpose() * q_old).squaredNorm();
        */

      }

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
  detectionsMutex.lock();

  for (auto it = attachedMarkers.begin(); it != attachedMarkers.end(); it++){
    bool all_present = true;
    MatrixXd pts = MatrixXd(it->marker_ids.size(), 3);
    int i=0;
    for (auto marker_id = it->marker_ids.begin(); marker_id != it->marker_ids.end(); marker_id++){
      if (*marker_id < 0 || *marker_id >= msg->number_rigid_bodies || msg->z[*marker_id] <= -20000 ||
          msg->z[*marker_id] >= -500){
        all_present = false;
        break;
      } else {
        pts(i, 0) = msg->x[*marker_id] / 1000.0;
        pts(i, 1) = msg->y[*marker_id] / 1000.0;
        pts(i, 2) = msg->z[*marker_id] / 1000.0;
        i++;
      }
    }

    if (all_present){
      it->last_received = getUnixTime();

      Vector3d avg_pos = pts.colwise().sum() / (double)it->marker_ids.size();
      it->last_offset = avg_pos;

      // if we have enough info to get an orientation out as well...
      if (it->have_normal && it->marker_ids.size() >= 3){
        // do a plane fit
        pts.rowwise() -= avg_pos.transpose();
        JacobiSVD<MatrixXd> svd(pts.transpose(), ComputeThinU);
        it->last_normal = svd.matrixU().block<3, 1>(0, 2);
        // We want a normal that faces the camera (to resolve the ambiguity here)
        // optotrack camera "forward" is -z 
        if (it->last_normal.transpose() * Vector3d(0, 0, 1.) < 0.0)
          it->last_normal *= -1.0;
        it->have_last_normal = true;
      } else {
        it->have_last_normal = false;
      }
    }
  }



  detectionsMutex.unlock();
}
