#undef NDEBUG
#include "NonpenetratingObjectCost.hpp"

#include <assert.h> 
#include <fstream>
#include "common.hpp"
#include "drake/util/convexHull.h"
#include <cmath>
#include "sdf_2d_functions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "drake/systems/plants/joints/RevoluteJoint.h"

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

NonpenetratingObjectCost::NonpenetratingObjectCost(std::shared_ptr<RigidBodyTree> robot_, std::vector<int> robot_correspondences_,
        std::shared_ptr<RigidBodyTree> robot_object_, std::vector<int> robot_object_correspondences_, std::shared_ptr<lcm::LCM> lcm_, YAML::Node config) :
    lcm(lcm_),
    robot(robot_),
    robot_kinematics_cache(robot->bodies),
    nq(robot->number_of_positions()),
    robot_correspondences(robot_correspondences_),
    robot_object(robot_object_),
    robot_object_kinematics_cache(robot_object->bodies),
    nq_object(robot_object->number_of_positions()),
    robot_object_correspondences(robot_object_correspondences_)
{
  std::cout << "Important #1: " << robot_correspondences.size() << "\n";
  std::cout << "Important #2: " << robot_object_correspondences.size() << "\n";

  std::cout << "\n";
  for (int i=0; i<robot_correspondences.size(); i++)
    std::cout << robot_correspondences[i] << " ";
  std::cout << "\n";
  for (int i=0; i<robot_object_correspondences.size(); i++)
    std::cout << robot_object_correspondences[i] << " ";
  std::cout << "\n";std::cout << "\n";

  if (config["nonpenetration_var"])
    nonpenetration_var = config["nonpenetration_var"].as<double>();
  if (config["verbose"])
    verbose = config["verbose"].as<bool>();
  if (config["verbose_lcmgl"])
    verbose_lcmgl = config["verbose_lcmgl"].as<bool>();
  if (config["num_surface_pts"])
    num_surface_pts = config["num_surface_pts"].as<int>();
  if (config["timeout_time"])
    timeout_time = config["timeout_time"].as<double>();


  robot_object_id = 1; //robot_object->FindBodyIndex("body");

  //lcmgl_lidar_= bot_lcmgl_init(lcm->getUnderlyingLCM(), "trimmed_lidar");
  //lcmgl_icp_= bot_lcmgl_init(lcm->getUnderlyingLCM(), "icp_p2pl");
  //lcmgl_measurement_model_ = bot_lcmgl_init(lcm->getUnderlyingLCM(), "meas_model");

  cv::namedWindow( "NonpenetratingObjectCostDebug", cv::WINDOW_AUTOSIZE );
  cv::startWindowThread();


  lcmgl_surface_pts_ = bot_lcmgl_init(lcm->getUnderlyingLCM(), "object_surface_pts");
  lcmgl_nonpen_corresp_ = bot_lcmgl_init(lcm->getUnderlyingLCM(), "nonpen_corresp");

  //auto kinect_frame_sub = lcm->subscribe("KINECT_FRAME", &NonpenetratingObjectCost::handleKinectFrameMsg, this);
  //kinect_frame_sub->setQueueCapacity(1);

  //uto save_pc_sub = lcm->subscribe("IRB140_ESTIMATOR_SAVE_POINTCLOUD", &NonpenetratingObjectCost::handleSavePointcloudMsg, this);
  //save_pc_sub->setQueueCapacity(1);

  VectorXd q_object_old(robot_object->number_of_positions());
  q_object_old *= 0;
  robot_object_kinematics_cache.initialize(q_object_old);
  robot_object->doKinematics(robot_object_kinematics_cache);

    // Use raycasting on robot_object to get a smattering of points on object surface
  surface_pts.resize(3, num_surface_pts);

  int num_good_surface_pts = 0;
  while(num_good_surface_pts < num_surface_pts){
    int attempt_num_pts = num_surface_pts - num_good_surface_pts;
    Matrix3Xd source_pts(3, attempt_num_pts);
    Matrix3Xd dest_pts(3, attempt_num_pts);
    double width = 0.1;
    for (int i=0; i<attempt_num_pts; i++) {
      source_pts.block<3,1>(0,i) = 
                     width * Vector3d((2.0*((double)rand())/RAND_MAX) - 1.0, 
                              (2.0*((double)rand())/RAND_MAX) - 1.0, 
                              (2.0*((double)rand())/RAND_MAX) - 1.0);
      dest_pts.block<3,1>(0,i) = 
                     width * Vector3d((2.0*((double)rand())/RAND_MAX) - 1.0, 
                              (2.0*((double)rand())/RAND_MAX) - 1.0, 
                              (2.0*((double)rand())/RAND_MAX) - 1.0);
    }

    // get them into object frame
    source_pts = robot_object->transformPoints(robot_object_kinematics_cache, source_pts, robot_object_id, 0);
    dest_pts = robot_object->transformPoints(robot_object_kinematics_cache, dest_pts, robot_object_id, 0);

    Eigen::VectorXd distances;
    Eigen::Matrix3Xd normals;
    std::vector<int> body_ids;
    robot_object->collisionRaycast(robot_object_kinematics_cache,
                          source_pts,
                          dest_pts,
                          distances, normals,
                          body_ids,
                          false);

    for (int i=0; i < attempt_num_pts; i++){
      if (distances(i) > 0){
        VectorXd between = (dest_pts.block<3,1>(0, i) - source_pts.block<3,1>(0, i));
        between /= between.norm();
        surface_pts.block<3, 1>(0, num_good_surface_pts) = source_pts.block<3, 1>(0, i) + distances(i) * between;  
        num_good_surface_pts++;
        if (num_good_surface_pts == num_surface_pts)
          break;
      }
    }
  }

  surface_pts = robot_object->transformPoints(robot_object_kinematics_cache, surface_pts, 0, robot_object_id);

  lastReceivedTime = getUnixTime() - timeout_time*2.;
}

void NonpenetratingObjectCost::initBotConfig(const char* filename)
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

int NonpenetratingObjectCost::get_trans_with_utime(std::string from_frame, std::string to_frame,
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

/***********************************************
            KNOWN POSITION HINTS
*********************************************/
bool NonpenetratingObjectCost::constructCost(ManipulationTracker * tracker, const Eigen::Matrix<double, Eigen::Dynamic, 1> x_old, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Q, Eigen::Matrix<double, Eigen::Dynamic, 1>& f, double& K)
{
  double now = getUnixTime();

  if (now - lastReceivedTime > timeout_time){
    if (verbose)
      printf("NonpenetratingObjectCost: constructed but timed out\n");
    //return false;
  }


  // TODO: LOOOOOOOTS TO DO HERE

  int nq_full = tracker->getRobot()->number_of_positions();
  VectorXd q_old_full = x_old.block(0,0,nq_full, 1);

  // First, convert x_old into corresponding q values for robot and robot_object
  VectorXd q_old(robot->number_of_positions());
  for (int i=0; i<robot_correspondences.size(); i++)
    q_old(i) = x_old(robot_correspondences[i]);
  robot_kinematics_cache.initialize(q_old);
  robot->doKinematics(robot_kinematics_cache);

  VectorXd q_object_old(robot_object->number_of_positions());
  for (int i=0; i<robot_object_correspondences.size(); i++)
    q_object_old(i) = x_old(robot_object_correspondences[i]);
  robot_object_kinematics_cache.initialize(q_object_old);
  robot_object->doKinematics(robot_object_kinematics_cache);
  

  Matrix3Xd global_surface_pts = robot_object->transformPoints(robot_object_kinematics_cache, surface_pts, robot_object_id, 0);  

  if (!std::isinf(nonpenetration_var) && global_surface_pts.cols() > 0){
    double NONPENETRATION_WEIGHT = 1. / (2. * nonpenetration_var * nonpenetration_var);
    
    VectorXd phi(global_surface_pts.cols());
    Matrix3Xd normal(3, global_surface_pts.cols()), x(3, global_surface_pts.cols()), body_x(3, global_surface_pts.cols());
    std::vector<int> body_idx(global_surface_pts.cols());
    // project points onto the collide-with object surfaces
    // via the last state estimate
    double now1 = getUnixTime();
    robot->collisionDetectFromPoints(robot_kinematics_cache, global_surface_pts,
                         phi, normal, x, body_x, body_idx, false);
    if (verbose)
      printf("Nonpenetration Contact Points SDF took %f\n", getUnixTime()-now1);

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
            assert(j < global_surface_pts.cols());
            if (global_surface_pts(0, j) == 0.0){
              cout << "Zero points " << global_surface_pts.block<3, 1>(0, j).transpose() << " slipping in at bdyidx " << body_idx[j] << endl;
            }
            if (phi(j) < -min_considered_penetration_distance){
              z.block<3, 1>(0, k) = global_surface_pts.block<3, 1>(0, j);
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
        MatrixXd J_z = robot_object->transformPointsJacobian(robot_object_kinematics_cache, z, robot_object_id, 0, false);
        MatrixXd J(3*z.cols(), nq_full);
        J.setZero();

        for (int j=0; j < nq; j++){
          J.block(0, robot_correspondences[j], 3*z.cols(), 1) += J_prime.block(0, j, 3*z.cols(), 1);
        }
        for (int j=0; j < nq_object; j++){
          J.block(0, robot_object_correspondences[j], 3*z.cols(), 1) -= J_z.block(0, j, 3*z.cols(), 1);
        }

        // minimize distance between the given set of points on the sensor surface,
        // and the given set of points on the body surface
        // min_{q_new} [ z - z_prime ]
        // min_{q_new} [ (z + J_z*(q_new - q_old)) - (z_prime + J_prime*(q_new - q_old)) ]
        // min_{q_new} [ (z - z_prime) + (J_z - J_prime)*(q_new - q_old) ]
        //NONPENETRATION_WEIGHT = 0.0;
        if (verbose_lcmgl){
          bot_lcmgl_begin(lcmgl_nonpen_corresp_, LCMGL_LINES);
          bot_lcmgl_line_width(lcmgl_nonpen_corresp_, 4.0f);   
          bot_lcmgl_color3f(lcmgl_nonpen_corresp_, 1.0, 0.0, 0.0);
        }

        for (int j=0; j < k; j++){
          MatrixXd Ks = z.col(j) - z_prime.col(j) + J.block(3*j, 0, 3, nq_full)*q_old_full;
          f.block(0, 0, nq_full, 1) -= NONPENETRATION_WEIGHT*(2. * Ks.transpose() * J.block(3*j, 0, 3, nq_full)).transpose()/(double)k;
          Q.block(0, 0, nq_full, nq_full) += NONPENETRATION_WEIGHT*(2. *  J.block(3*j, 0, 3, nq_full).transpose() * J.block(3*j, 0, 3, nq_full))/(double)k;
          K += NONPENETRATION_WEIGHT*Ks.squaredNorm()/(double)k;

          if (verbose_lcmgl && j % 1 == 0){
            // visualize point correspondences and normals
            double dist_normalized = fmin(0.05, (z.col(j) - z_prime.col(j)).norm()) / 0.05;
          //  bot_lcmgl_color3f(lcmgl_nonpen_corresp_, 1.0, 0.0, (1.0-dist_normalized)*(1.0-dist_normalized));
            bot_lcmgl_vertex3f(lcmgl_nonpen_corresp_, z(0, j), z(1, j), z(2, j));
            //Vector3d norm_endpt = z_prime.block<3,1>(0,j) + z_norms.block<3,1>(0,j)*0.01;
            //bot_lcmgl_vertex3f(lcmgl_nonpen_corresp_, norm_endpt(0), norm_endpt(1), norm_endpt(2));
            bot_lcmgl_vertex3f(lcmgl_nonpen_corresp_, z_prime(0, j), z_prime(1, j), z_prime(2, j));
            
          }
        }
        if (verbose_lcmgl)
          bot_lcmgl_end(lcmgl_nonpen_corresp_);  
      }
    }
  }
    

  // draw them for debug
  if (verbose_lcmgl) {
    bot_lcmgl_switch_buffer(lcmgl_nonpen_corresp_);  

    bot_lcmgl_point_size(lcmgl_surface_pts_, 4.0f);
    bot_lcmgl_begin(lcmgl_surface_pts_, LCMGL_POINTS);

    bot_lcmgl_color3f(lcmgl_surface_pts_, 0, 1, 0);  
    for (int i=0; i < global_surface_pts.cols(); i++){
      bot_lcmgl_vertex3f(lcmgl_surface_pts_, global_surface_pts(0, i), global_surface_pts(1, i), global_surface_pts(2, i));
    }
    bot_lcmgl_color3f(lcmgl_surface_pts_, 1, 0, 0);
    bot_lcmgl_end(lcmgl_surface_pts_);

    bot_lcmgl_switch_buffer(lcmgl_surface_pts_);
  }
  
  // Downsample object surface points

  // Loop over points, colliding them with robot() geometry to find nearest-point correspondences

  // Proceed as usual... ICP

  return true;

}
