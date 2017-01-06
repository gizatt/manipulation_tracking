#undef NDEBUG
#include "GelsightCost.hpp"

#include <assert.h> 
#include <fstream>
#include "common/common.hpp"
#include "drake/util/convexHull.h"
#include <cmath>
#include "common/sdf_2d_functions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace Eigen;
using namespace cv;

template<typename _Tp, int _rows, int _cols, int _options, int _maxRows, int _maxCols>
void eigen2cv( const Eigen::Matrix<_Tp, _rows, _cols, _options, _maxRows, _maxCols>& src, cv::Mat& dst)
{
    if( !(src.Flags & Eigen::RowMajorBit) )
    {
        cv::Mat _src(src.cols(), src.rows(), cv::DataType<_Tp>::type,
              (void*)src.data(), src.stride()*sizeof(_Tp));
        transpose(_src, dst);
    }
    else
    {
        cv::Mat _src(src.rows(), src.cols(), cv::DataType<_Tp>::type,
                 (void*)src.data(), src.stride()*sizeof(_Tp));
        _src.copyTo(dst);
    }
}

GelsightCost::GelsightCost(std::shared_ptr<RigidBodyTree<double> > robot_, std::shared_ptr<lcm::LCM> lcm_, YAML::Node config) :
    robot(robot_),
    robot_kinematics_cache(robot->get_num_positions(), robot->get_num_velocities()),
    lcm(lcm_),
    nq(robot->get_num_positions())
{
  if (config["downsample_amount"])
    downsample_amount = config["downsample_amount"].as<double>();
  if (config["gelsight_freespace_var"])
    gelsight_freespace_var = config["gelsight_freespace_var"].as<double>();
  if (config["gelsight_depth_var"])
    gelsight_depth_var = config["gelsight_depth_var"].as<double>();
  if (config["timeout_time"])
    timeout_time = config["timeout_time"].as<double>();
  if (config["contact_threshold"])
    contact_threshold = config["contact_threshold"].as<double>();
  if (config["max_considered_corresp_distance"])
    max_considered_corresp_distance = config["max_considered_corresp_distance"].as<double>();
  if (config["min_considered_penetration_distance"])
    min_considered_penetration_distance = config["min_considered_penetration_distance"].as<double>();
  if (config["verbose"])
    verbose = config["verbose"].as<bool>();

  if (config["surface"]){
    sensor_plane.lower_left = Vector3d(config["surface"]["lower_left"][0].as<double>(), 
                                       config["surface"]["lower_left"][1].as<double>(), 
                                       config["surface"]["lower_left"][2].as<double>());
    sensor_plane.lower_right = Vector3d(config["surface"]["lower_right"][0].as<double>(), 
                                       config["surface"]["lower_right"][1].as<double>(), 
                                       config["surface"]["lower_right"][2].as<double>());
    sensor_plane.upper_left = Vector3d(config["surface"]["upper_left"][0].as<double>(), 
                                       config["surface"]["upper_left"][1].as<double>(), 
                                       config["surface"]["upper_left"][2].as<double>());
    sensor_plane.upper_right = Vector3d(config["surface"]["upper_right"][0].as<double>(), 
                                       config["surface"]["upper_right"][1].as<double>(), 
                                       config["surface"]["upper_right"][2].as<double>());
    sensor_plane.normal = Vector3d(config["surface"]["normal"][0].as<double>(), 
                                   config["surface"]["normal"][1].as<double>(), 
                                   config["surface"]["normal"][2].as<double>());
    sensor_plane.thickness = config["surface"]["thickness"].as<double>();
    if (config["surface"]["body"])
      sensor_body_id = robot->FindBodyIndex(config["surface"]["body"].as<string>());
    else
      sensor_body_id = 0; // default to world... this puts gelsight at the origin
  } else {
    printf("Must define image plane to use a Gelsight cost\n");
    exit(1);
  }


 // don't know why this doesn't work. for now, just gonna use URDF with no 
 // collision...
  // remove collision geometry from gelsight collision group
  /*
  auto filter = [&](const std::string& group_name) {
    return group_name == std::string("gelsight");
  };
  robot->removeCollisionGroupsIf(filter);
  robot->compile();
  */

  lcmgl_gelsight_ = bot_lcmgl_init(lcm->getUnderlyingLCM(), "gelsight");
  lcmgl_corresp_ = bot_lcmgl_init(lcm->getUnderlyingLCM(), "gelsight_corrs");

  latest_gelsight_image.resize(input_num_pixel_rows, input_num_pixel_cols);

  cv::namedWindow( "GelsightDepth", cv::WINDOW_AUTOSIZE );
  cv::startWindowThread();

  auto gelsight_frame_sub = lcm->subscribe("GELSIGHT_DEPTH", &GelsightCost::handleGelsightFrameMsg, this);
  gelsight_frame_sub->setQueueCapacity(1);

  lastReceivedTime = getUnixTime() - timeout_time*2.;
  startTime = getUnixTime();
}

/***********************************************
            KNOWN POSITION HINTS
*********************************************/
bool GelsightCost::constructCost(ManipulationTracker * tracker, const Eigen::Matrix<double, Eigen::Dynamic, 1> x_old, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Q, Eigen::Matrix<double, Eigen::Dynamic, 1>& f, double& K)
{
  double now = getUnixTime();
  if (now - lastReceivedTime > timeout_time || now - startTime < 1.0){ // slight delay helps when restarting tracker during contact...
    if (verbose)
      printf("GelsightCost: constructed but timed out\n");
    return false;
  }
  else {
    VectorXd q_old = x_old.block(0, 0, robot->get_num_positions(), 1);
    robot_kinematics_cache.initialize(q_old);
    robot->doKinematics(robot_kinematics_cache);

    cv::Mat image_disp;
    Eigen::MatrixXd gelsight_image;
    gelsight_frame_mutex.lock();
    eigen2cv(latest_gelsight_image, image_disp);
    gelsight_image.resize(latest_gelsight_image.rows(), latest_gelsight_image.cols());
    gelsight_image = latest_gelsight_image;
    gelsight_frame_mutex.unlock();
    if (latest_gelsight_image.size() > 0){
      cv::resize(image_disp, image_disp, cv::Size(640, 480));
      cv::imshow("GelsightDepth", image_disp);
    }


    // sample pixels from each downsample region
    // and sort into positive, negative points 
    Matrix3Xd contact_points(3, num_pixel_rows*num_pixel_cols);
    Matrix3Xd noncontact_points(3, num_pixel_rows*num_pixel_cols);
    int num_contact_points = 0;
    int num_noncontact_points = 0;

    for (size_t v=0; v<num_pixel_rows; v++) {
      for (size_t u=0; u<num_pixel_cols; u++) {
        int full_v = min((int)floor(((double)v)*downsample_amount) + rand()%(int)downsample_amount, input_num_pixel_rows-1);
        int full_u = min((int)floor(((double)u)*downsample_amount) + rand()%(int)downsample_amount, input_num_pixel_cols-1);

        Vector3d this_point = sensor_plane.lower_left + (sensor_plane.lower_right-sensor_plane.lower_left)*(((double)full_u) / (double)input_num_pixel_cols)
                           + (sensor_plane.upper_left-sensor_plane.lower_left)*(((double)full_v) / (double)input_num_pixel_rows);

        double val = gelsight_image(full_v, full_u);
        if (val >= contact_threshold){
          contact_points.block<3,1>(0,num_contact_points) = this_point + (1.0-val) * sensor_plane.thickness * sensor_plane.normal;
          num_contact_points++;
        } else {
          noncontact_points.block<3,1>(0,num_noncontact_points) = this_point + (1.0-val) * sensor_plane.thickness * sensor_plane.normal;
          num_noncontact_points++; 
        }
      }
    }

    contact_points.conservativeResize(3, num_contact_points);
    noncontact_points.conservativeResize(3, num_noncontact_points);

    // get them into sensor frame
    contact_points = robot->transformPoints(robot_kinematics_cache, contact_points, sensor_body_id, 0);
    noncontact_points = robot->transformPoints(robot_kinematics_cache, noncontact_points, sensor_body_id, 0);

    // draw them for debug

    
    bot_lcmgl_point_size(lcmgl_gelsight_, 4.0f);
    bot_lcmgl_begin(lcmgl_gelsight_, LCMGL_POINTS);

    bot_lcmgl_color3f(lcmgl_gelsight_, 0, 1, 0);  
    for (int i=0; i < contact_points.cols(); i++){
      bot_lcmgl_vertex3f(lcmgl_gelsight_, contact_points(0, i), contact_points(1, i), contact_points(2, i));
    }
    bot_lcmgl_color3f(lcmgl_gelsight_, 1, 0, 0);  
    for (int i=0; i < noncontact_points.cols(); i++){
      bot_lcmgl_vertex3f(lcmgl_gelsight_, noncontact_points(0, i), noncontact_points(1, i), noncontact_points(2, i));
    }
    bot_lcmgl_end(lcmgl_gelsight_);
    


    // for each point for which we have contact, attract nearby surfaces
    if (!std::isinf(gelsight_depth_var) && contact_points.cols() > 0){
      double DEPTH_WEIGHT = 1. / (2. * gelsight_depth_var * gelsight_depth_var);
      
      VectorXd phi(contact_points.cols());
      Matrix3Xd normal(3, contact_points.cols()), x(3, contact_points.cols()), body_x(3, contact_points.cols());
      std::vector<int> body_idx(contact_points.cols());
      // project points onto object surfaces (minus gelsight collision group)
      // via the last state estimate
      double now1 = getUnixTime();
      robot->collisionDetectFromPoints(robot_kinematics_cache, contact_points,
                           phi, normal, x, body_x, body_idx, false);
      if (verbose)
        printf("Gelsight Contact Points SDF took %f\n", getUnixTime()-now1);

      // for every unique body points have returned onto...
      std::vector<int> num_points_on_body(robot->bodies.size(), 0);
      for (int i=0; i < body_idx.size(); i++)
        num_points_on_body[body_idx[i]] += 1;

      // for every body...
      for (int i=0; i < robot->bodies.size(); i++){
        if (num_points_on_body[i] > 0){

          // collect results from raycast that correspond to this body out in the world
          Matrix3Xd z(3, num_points_on_body[i]); // points, in world frame, near this body
          Matrix3Xd z_prime(3, num_points_on_body[i]); // same points projected onto surface of body
          Matrix3Xd body_z_prime(3, num_points_on_body[i]); // projected points in body frame
          Matrix3Xd z_norms(3, num_points_on_body[i]); // normals corresponding to these points
          int k = 0;
          for (int j=0; j < body_idx.size(); j++){
            assert(k < body_idx.size());
            if (body_idx[j] == i){
              assert(j < contact_points.cols());
              if (contact_points(0, j) == 0.0){
                cout << "Zero points " << contact_points.block<3, 1>(0, j).transpose() << " slipping in at bdyidx " << body_idx[j] << endl;
              }
              if ((contact_points.block<3, 1>(0, j) - x.block<3, 1>(0, j)).norm() <= max_considered_corresp_distance){
                z.block<3, 1>(0, k) = contact_points.block<3, 1>(0, j);
                z_prime.block<3, 1>(0, k) = x.block<3, 1>(0, j);
                body_z_prime.block<3, 1>(0, k) = body_x.block<3, 1>(0, j);
                z_norms.block<3, 1>(0, k) = normal.block<3, 1>(0, j);
                k++;
              }
            }
          }

          z.conservativeResize(3, k);
          z_prime.conservativeResize(3, k);
          body_z_prime.conservativeResize(3, k);
          z_norms.conservativeResize(3, k);

          // forwardkin to get our jacobians on the body we're currently iterating on, as well as from
          // the sensor body id
          MatrixXd J_prime = robot->transformPointsJacobian(robot_kinematics_cache, body_z_prime, i, 0, false);
          MatrixXd J_z = robot->transformPointsJacobian(robot_kinematics_cache, z, sensor_body_id, 0, false);
          MatrixXd J = J_prime - J_z;

          // minimize distance between the given set of points on the sensor surface,
          // and the given set of points on the body surface
          // min_{q_new} [ z - z_prime ]
          // min_{q_new} [ (z + J_z*(q_new - q_old)) - (z_prime + J_prime*(q_new - q_old)) ]
          // min_{q_new} [ (z - z_prime) + (J_z - J_prime)*(q_new - q_old) ]
          bot_lcmgl_begin(lcmgl_corresp_, LCMGL_LINES);
          bot_lcmgl_line_width(lcmgl_corresp_, 4.0f);
          bot_lcmgl_color3f(lcmgl_corresp_, 0.0, 0.0, 1.0);
              
          for (int j=0; j < k; j++){
            MatrixXd Ks = z.col(j) - z_prime.col(j) + J.block(3*j, 0, 3, nq)*q_old;
            f.block(0, 0, nq, 1) -= DEPTH_WEIGHT*(2. * Ks.transpose() * J.block(3*j, 0, 3, nq)).transpose()/(double)k;
            Q.block(0, 0, nq, nq) += DEPTH_WEIGHT*(2. *  J.block(3*j, 0, 3, nq).transpose() * J.block(3*j, 0, 3, nq))/(double)k;
            K += DEPTH_WEIGHT*Ks.squaredNorm()/(double)k;

            if (j % 1 == 0){
              // visualize point correspondences and normals
              double dist_normalized = fmin(max_considered_corresp_distance, (z.col(j) - z_prime.col(j)).norm()) / max_considered_corresp_distance;
              //bot_lcmgl_color3f(lcmgl_corresp_, dist_normalized*dist_normalized, 0, (1.0-dist_normalized)*(1.0-dist_normalized));
              bot_lcmgl_vertex3f(lcmgl_corresp_, z(0, j), z(1, j), z(2, j));
              bot_lcmgl_vertex3f(lcmgl_corresp_, z_prime(0, j), z_prime(1, j), z_prime(2, j));
              
            }
          }
          bot_lcmgl_end(lcmgl_corresp_);  
        }
      }
    }


    //ghost of an issue where this cost causes the object to get dragged if it gets
    //significantly behind the gelsight surface, somehow? very weird.

    // and for each point for which we have no contact, repel nearby surfaces
    if (!std::isinf(gelsight_freespace_var) && noncontact_points.cols() > 0){
      double FREESPACE_WEIGHT = 1. / (2. * gelsight_freespace_var * gelsight_freespace_var);
      
      VectorXd phi(noncontact_points.cols());
      Matrix3Xd normal(3, noncontact_points.cols()), x(3, noncontact_points.cols()), body_x(3, noncontact_points.cols());
      std::vector<int> body_idx(noncontact_points.cols());
      // project points onto object surfaces (minus gelsight collision group)
      // via the last state estimate
      double now1 = getUnixTime();
      robot->collisionDetectFromPoints(robot_kinematics_cache, noncontact_points,
                           phi, normal, x, body_x, body_idx, false);
      if (verbose)
        printf("Gelsight Contact Points SDF took %f\n", getUnixTime()-now1);

      // for every unique body points have returned onto...
      std::vector<int> num_points_on_body(robot->bodies.size(), 0);
      for (int i=0; i < body_idx.size(); i++)
        num_points_on_body[body_idx[i]] += 1;

      // for every body...
      for (int i=0; i < robot->bodies.size(); i++){
        if (num_points_on_body[i] > 0){

          // collect results from raycast that correspond to this body out in the world
          VectorXd phis(num_points_on_body[i]);
          Matrix3Xd z(3, num_points_on_body[i]); // points, in world frame, near this body
          Matrix3Xd z_prime(3, num_points_on_body[i]); // same points projected onto surface of body
          Matrix3Xd body_z_prime(3, num_points_on_body[i]); // projected points in body frame
          Matrix3Xd z_norms(3, num_points_on_body[i]); // normals corresponding to these points
          int k = 0;

          for (int j=0; j < body_idx.size(); j++){
            assert(k < body_idx.size());
            if (body_idx[j] == i){
              assert(j < noncontact_points.cols());
              if (noncontact_points(0, j) == 0.0){
                cout << "Zero points " << noncontact_points.block<3, 1>(0, j).transpose() << " slipping in at bdyidx " << body_idx[j] << endl;
              }
              if (phi(j) < -min_considered_penetration_distance){
                z.block<3, 1>(0, k) = noncontact_points.block<3, 1>(0, j);
                z_prime.block<3, 1>(0, k) = x.block<3, 1>(0, j);
                body_z_prime.block<3, 1>(0, k) = body_x.block<3, 1>(0, j);
                z_norms.block<3, 1>(0, k) = normal.block<3, 1>(0, j);
                phis(k) = phi(j);
                k++;
              }
            }
          }

          z.conservativeResize(3, k);
          z_prime.conservativeResize(3, k);
          body_z_prime.conservativeResize(3, k);
          z_norms.conservativeResize(3, k);
          phis.conservativeResize(k);

          // forwardkin to get our jacobians on the body we're currently iterating on, as well as from
          // the sensor body id
          MatrixXd J_prime = robot->transformPointsJacobian(robot_kinematics_cache, body_z_prime, i, 0, false);
          MatrixXd J_z = robot->transformPointsJacobian(robot_kinematics_cache, z, sensor_body_id, 0, false);
          MatrixXd J = J_prime - J_z;

          // minimize distance between the given set of points on the sensor surface,
          // and the given set of points on the body surface
          // min_{q_new} [ z - z_prime ]
          // min_{q_new} [ (z + J_z*(q_new - q_old)) - (z_prime + J_prime*(q_new - q_old)) ]
          // min_{q_new} [ (z - z_prime) + (J_z - J_prime)*(q_new - q_old) ]
          //FREESPACE_WEIGHT = 0.0;
          bot_lcmgl_begin(lcmgl_corresp_, LCMGL_LINES);
          bot_lcmgl_line_width(lcmgl_corresp_, 4.0f);   
          bot_lcmgl_color3f(lcmgl_corresp_, 1.0, 0.0, 0.0);

          for (int j=0; j < k; j++){
            MatrixXd Ks = z.col(j) - z_prime.col(j) + J.block(3*j, 0, 3, nq)*q_old;
            f.block(0, 0, nq, 1) -= FREESPACE_WEIGHT*(2. * Ks.transpose() * J.block(3*j, 0, 3, nq)).transpose()/(double)k;
            Q.block(0, 0, nq, nq) += FREESPACE_WEIGHT*(2. *  J.block(3*j, 0, 3, nq).transpose() * J.block(3*j, 0, 3, nq))/(double)k;
            K += FREESPACE_WEIGHT*Ks.squaredNorm()/(double)k;

            if (j % 1 == 0){
              // visualize point correspondences and normals
              double dist_normalized = fmin(max_considered_corresp_distance, (z.col(j) - z_prime.col(j)).norm()) / max_considered_corresp_distance;
            //  bot_lcmgl_color3f(lcmgl_corresp_, 1.0, 0.0, (1.0-dist_normalized)*(1.0-dist_normalized));
              bot_lcmgl_vertex3f(lcmgl_corresp_, z(0, j), z(1, j), z(2, j));
              //Vector3d norm_endpt = z_prime.block<3,1>(0,j) + z_norms.block<3,1>(0,j)*0.01;
              //bot_lcmgl_vertex3f(lcmgl_corresp_, norm_endpt(0), norm_endpt(1), norm_endpt(2));
              bot_lcmgl_vertex3f(lcmgl_corresp_, z_prime(0, j), z_prime(1, j), z_prime(2, j));
              
            }
          }
          bot_lcmgl_end(lcmgl_corresp_);  
        }
      }
      bot_lcmgl_switch_buffer(lcmgl_corresp_);  

    }

    bot_lcmgl_switch_buffer(lcmgl_gelsight_);  

    if (verbose)
      printf("Spend %f in gelsight constraints.\n", getUnixTime() - now);

    return true;
  }
}

void GelsightCost::updateGelsightImage(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> new_gelsight_image){
  gelsight_frame_mutex.lock();
  latest_gelsight_image = new_gelsight_image;
  input_num_pixel_cols = latest_gelsight_image.cols();
  input_num_pixel_rows = latest_gelsight_image.rows();

  num_pixel_cols = (int) floor( ((double)input_num_pixel_cols) / downsample_amount);
  num_pixel_rows = (int) floor( ((double)input_num_pixel_rows) / downsample_amount);

  gelsight_frame_mutex.unlock();
  lastReceivedTime = getUnixTime();
}

void GelsightCost::handleGelsightFrameMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::image_t* msg){
  //printf("Received gelsight frame on channel %s\n", chan.c_str());

  gelsight_frame_mutex.lock();
 

  // gelsight frame comes in as an image
  bool success = false;
  cv::Mat decodedImage;
  if (msg->pixelformat == bot_core::image_t::PIXEL_FORMAT_MJPEG){
   decodedImage = cv::imdecode(msg->data, 0);
   success = (decodedImage.rows > 0);

   if (success){
    if (latest_gelsight_image.rows() != msg->height || latest_gelsight_image.cols() != msg->width){
      latest_gelsight_image.resize(msg->height, msg->width);
      input_num_pixel_rows = msg->height;
      input_num_pixel_cols = msg->width;
    }
    for(long int v=0; v<input_num_pixel_rows; v++) { // t2b self->height 480
      for(long int u=0; u<input_num_pixel_cols; u++ ) {  //l2r self->width 640
        latest_gelsight_image(v, u) = ((float) decodedImage.at<uint8_t>(v, u)) / 255.0;
      }
    }
   }

  } else if (msg->pixelformat == bot_core::image_t::PIXEL_FORMAT_GRAY){
    if (latest_gelsight_image.rows() != msg->height || latest_gelsight_image.cols() != msg->width){
      latest_gelsight_image.resize(msg->height, msg->width);
      input_num_pixel_rows = msg->height;
      input_num_pixel_cols = msg->width;
    }
    for(long int v=0; v<input_num_pixel_rows; v++) { // t2b self->height 480
      for(long int u=0; u<input_num_pixel_cols; u++ ) {  //l2r self->width 640
        latest_gelsight_image(v, u) = ((float) msg->data[v*input_num_pixel_cols+u]) / 255.0;
      }
    }
  } else {
   printf("Got a Gelsight image in a format I don't understand: %d\n", msg->pixelformat);
  }

  num_pixel_cols = (int) floor( ((double)input_num_pixel_cols) / downsample_amount);
  num_pixel_rows = (int) floor( ((double)input_num_pixel_rows) / downsample_amount);
  

  gelsight_frame_mutex.unlock();

  lastReceivedTime = getUnixTime();
}
