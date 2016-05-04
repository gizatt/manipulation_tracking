#undef NDEBUG
#include <assert.h> 
#include <fstream>
#include "IRB140Estimator.hpp"
#include "drake/util/convexHull.h"
#include "zlib.h"
#include "sdf_2d_functions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include "drake/systems/plants/joints/RevoluteJoint.h"

using namespace std;
using namespace Eigen;

#define MAX_SCAN_DIST 10.0

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

// from https://forum.kde.org/viewtopic.php?f=74&t=91514
template<typename Derived>
inline bool is_finite(const Eigen::MatrixBase<Derived>& x)
{
   return ( (x - x).array() == (x - x).array()).all();
}
template<typename Derived>
inline bool is_nan(const Eigen::MatrixBase<Derived>& x)
{
   return ((x.array() == x.array())).all();
}

IRB140Estimator::IRB140Estimator(std::shared_ptr<RigidBodyTree> arm, std::shared_ptr<RigidBodyTree> manipuland, 
      Eigen::Matrix<double, Eigen::Dynamic, 1> x0_arm, Eigen::Matrix<double, Eigen::Dynamic, 1> x0_manipuland,
    const char* filename, const char* state_channelname, bool transcribe_published_floating_base,
    const char* hand_state_channelname) :
    x_arm(x0_arm), x_manipuland(x0_manipuland), manipuland_kinematics_cache(manipuland->bodies),
    transcribe_published_floating_base(transcribe_published_floating_base)
{
  this->arm = arm;
  this->manipuland = manipuland;

  last_update_time = getUnixTime();

  if (!this->lcm.good()) {
    throw std::runtime_error("LCM is not good");
  }

  lcmgl_lidar_= bot_lcmgl_init(lcm.getUnderlyingLCM(), "trimmed_lidar");
  lcmgl_manipuland_= bot_lcmgl_init(lcm.getUnderlyingLCM(), "manipuland_se");
  lcmgl_icp_= bot_lcmgl_init(lcm.getUnderlyingLCM(), "icp_p2pl");
  lcmgl_measurement_model_ = bot_lcmgl_init(lcm.getUnderlyingLCM(), "meas_model");

  this->initBotConfig(filename);

  // if we're using a kinect... (to be refactored)
  // This is in full agreement with Kintinuous: (calibrationAsus.yml)
  // NB: if changing this, it should be kept in sync
  kcal = kinect_calib_new();
  kcal->intrinsics_depth.fx = 528.01442863461716;//was 576.09757860;
  kcal->intrinsics_depth.cx = 320.0;
  kcal->intrinsics_depth.cy = 267.0;
  kcal->intrinsics_rgb.fx = 528.01442863461716;//576.09757860; ... 528 seems to be better, emperically, march 2015
  kcal->intrinsics_rgb.cx = 320.0;
  kcal->intrinsics_rgb.cy = 267.0;
  kcal->intrinsics_rgb.k1 = 0; // none given so far
  kcal->intrinsics_rgb.k2 = 0; // none given so far
  kcal->shift_offset = 1090.0;
  kcal->projector_depth_baseline = 0.075;
  //double rotation[9];
  double rotation[]={0.999999, -0.000796, 0.001256, 0.000739, 0.998970, 0.045368, -0.001291, -0.045367, 0.998970};
  double depth_to_rgb_translation[] ={ -0.015756, -0.000923, 0.002316};
  memcpy(kcal->depth_to_rgb_rot, rotation, 9*sizeof(double));
  memcpy(kcal->depth_to_rgb_translation, depth_to_rgb_translation  , 3*sizeof(double));

  // generate sample points for doing sensor simulation
  // todo: verify vals / figure out how to regenerate raycast endpoints when we 
  // receive depth images and know the resolution
  /*
  double half_pitch_fov = atan2(kcal->intrinsics_depth.cy, kcal->intrinsics_depth.fx);
  double half_yaw_fov = atan2(kcal->intrinsics_depth.cx, kcal->intrinsics_depth.fx);
  double min_pitch = -half_pitch_fov;
  double max_pitch = half_pitch_fov;
  double min_yaw = -half_yaw_fov;
  double max_yaw = half_yaw_fov;
  */

  num_pixel_cols = (int) floor( ((double)input_num_pixel_cols) / downsample_amount);
  num_pixel_rows = (int) floor( ((double)input_num_pixel_rows) / downsample_amount);

  latest_depth_image.resize(input_num_pixel_rows, input_num_pixel_cols);
  raycast_endpoints.resize(3,num_pixel_rows*num_pixel_cols);

  x_manipuland_measured.resize(x0_manipuland.rows());
  x_manipuland_measured_known.resize(x0_manipuland.rows());
  fill(x_manipuland_measured_known.begin(), x_manipuland_measured_known.end(), false);

/*
  for (int i=0; i<12; i++){
    x_manipuland_measured_known[manipuland->num_positions-12+i] = true;
    x_manipuland_measured(manipuland->num_positions-12+i) = x_manipuland(manipuland->num_positions-12+i);
  }
*/


  this->setupSubscriptions(state_channelname, hand_state_channelname);

  //visualizer = make_shared<Drake::BotVisualizer<Drake::RigidBodySystem::StateVector>>(make_shared<lcm::LCM>(lcm),manipuland);

  cv::namedWindow( "IRB140EstimatorDebug", cv::WINDOW_AUTOSIZE );
  cv::startWindowThread();
}

void IRB140Estimator::initBotConfig(const char* filename)
{
  if (filename && filename[0])
    {
      botparam_ = bot_param_new_from_file(filename);
    }
  else
    {
    while (!botparam_)
      {
        botparam_ = bot_param_new_from_server(this->lcm.getUnderlyingLCM(), 0);
      }
    }
  botframes_ = bot_frames_get_global(this->lcm.getUnderlyingLCM(), botparam_);
}

int IRB140Estimator::get_trans_with_utime(std::string from_frame, std::string to_frame,
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

void IRB140Estimator::update(double dt){
  Eigen::Matrix3Xd full_cloud;
  Eigen::MatrixXd full_depth_image;
  latest_cloud_mutex.lock();
  full_cloud = latest_cloud;
  full_depth_image.resize(latest_depth_image.rows(), latest_depth_image.cols());
  full_depth_image= latest_depth_image;
  latest_cloud_mutex.unlock();

  VectorXd q_old = x_manipuland.block(0, 0, manipuland->num_positions, 1);
  manipuland_kinematics_cache.initialize(q_old);
  manipuland->doKinematics(manipuland_kinematics_cache);
  
  // transform into world frame
  Eigen::Isometry3d kinect2tag;
  long long utime = 0;
  this->get_trans_with_utime("KINECT_RGB", "KINECT_TO_APRILTAG", utime, kinect2tag);
  Eigen::Isometry3d world2tag;
  long long utime2 = 0;
  this->get_trans_with_utime("local", "robot_yplus_tag", utime2, world2tag);
  Eigen::Isometry3d kinect2world =  world2tag.inverse() * kinect2tag;
  full_cloud = kinect2world * full_cloud;

  // do randomized downsampling, populating data stores to be used by ICP
  Matrix3Xd points(3, full_cloud.cols()); int i=0;
  Eigen::MatrixXd depth_image; depth_image.resize(num_pixel_rows, num_pixel_cols);
  double constant = 1.0f / kcal->intrinsics_rgb.fx ;
  if (full_cloud.cols() > 0){
    if (full_cloud.cols() != input_num_pixel_cols*input_num_pixel_rows){
      printf("WARNING: SOMEHOW FULL CLOUD HAS WRONG NUMBER OF ENTRIES.\n");
    }
    for (size_t v=0; v<num_pixel_rows; v++) {
      for (size_t u=0; u<num_pixel_cols; u++) {
        int full_v = min((int)floor(((double)v)*downsample_amount) + rand()%(int)downsample_amount, input_num_pixel_rows);
        int full_u = min((int)floor(((double)u)*downsample_amount) + rand()%(int)downsample_amount, input_num_pixel_cols);

        // cut down to just point cloud in our manipulation space
        //(todo: bring in this info externally somehow)
        Eigen::Vector3d pt = full_cloud.block<3, 1>(0, full_v*input_num_pixel_cols + full_u);
        if (full_depth_image(full_v, full_u) > 0. &&
            pt[0] > manip_x_bounds[0] && pt[0] < manip_x_bounds[1] && 
            pt[1] > manip_y_bounds[0] && pt[1] < manip_y_bounds[1] && 
            pt[2] > manip_z_bounds[0] && pt[2] < manip_z_bounds[1]){
          assert(pt[0] != 0.0);
          points.block<3, 1>(0, i) = pt;
          i++;
        }

        // populate depth image using our random sample
        depth_image(v, u) = full_depth_image(full_v, full_u); 

        // populate raycast endpoints using our random sample
        raycast_endpoints.col(num_pixel_cols*v + u) = Vector3d(
          (((double) full_u)- kcal->intrinsics_depth.cx)*1.0*constant,
          (((double) full_v)- kcal->intrinsics_depth.cy)*1.0*constant, 
          1.0); // simulate the depth sensor;
        raycast_endpoints.col(num_pixel_cols*v + u) *= MAX_SCAN_DIST/(raycast_endpoints.col(num_pixel_cols*v + u).norm());
      }
    }
  }
  // conservativeResize keeps old coefficients
  // (regular resize would clear them)
  points.conservativeResize(3, i);


  double now=getUnixTime();
  this->performCompleteICP(kinect2world, depth_image, points);
  //printf("elapsed in articulated, constrainted ICP: %f\n", getUnixTime() - now);

  // visualize point cloud
  bot_lcmgl_point_size(lcmgl_lidar_, 4.5f);
  bot_lcmgl_color3f(lcmgl_lidar_, 0, 1, 0);
  
  bot_lcmgl_begin(lcmgl_lidar_, LCMGL_POINTS);
  for (i = 0; i < points.cols(); i++){
    if (i % 1 == 0) {
      bot_lcmgl_vertex3f(lcmgl_lidar_, points(0, i), points(1, i), points(2, i));
    }
  }
  bot_lcmgl_end(lcmgl_lidar_);
  bot_lcmgl_switch_buffer(lcmgl_lidar_);  

  // Publish the object state
  //cout << "Manipuland robot name vector: " << manipuland->robot_name.size() << endl;
  for (int roboti=1; roboti < manipuland->robot_name.size(); roboti++){
    bot_core::robot_state_t manipulation_state;
    manipulation_state.utime = getUnixTime();
    std::string robot_name = manipuland->robot_name[roboti];

    manipulation_state.num_joints = 0;
    bool found_floating = false;
    for (int i=0; i<manipuland->bodies.size(); i++){
      if (manipuland->bodies[i]->model_name == robot_name){
        if (manipuland->bodies[i]->getJoint().isFloating()){
          manipulation_state.pose.translation.x = x_manipuland[manipuland->bodies[i]->position_num_start + 0];
          manipulation_state.pose.translation.y = x_manipuland[manipuland->bodies[i]->position_num_start + 1];
          manipulation_state.pose.translation.z = x_manipuland[manipuland->bodies[i]->position_num_start + 2];
          auto quat = rpy2quat(x_manipuland.block<3, 1>(manipuland->bodies[i]->position_num_start + 3, 0));
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
          manipulation_state.num_joints += manipuland->bodies[i]->getJoint().getNumPositions();
          for (int j=0; j < manipuland->bodies[i]->getJoint().getNumPositions(); j++){
            manipulation_state.joint_name.push_back(manipuland->bodies[i]->getJoint().getPositionName(j));
            manipulation_state.joint_position.push_back(x_manipuland[manipuland->bodies[i]->position_num_start + j]);
            manipulation_state.joint_velocity.push_back(x_manipuland[manipuland->bodies[i]->position_num_start + j + manipuland->num_positions]);
          }
        }
      }
    }
    manipulation_state.joint_effort.resize(manipulation_state.num_joints, 0.0);
    std::string channelname = "EST_MANIPULAND_STATE_" + robot_name;
    //cout << " published to " << channelname << endl;
    lcm.publish(channelname, &manipulation_state);
  }
}  

void IRB140Estimator::performCompleteICP(Eigen::Isometry3d& kinect2world, Eigen::MatrixXd& depth_image, Eigen::Matrix3Xd& points){
  int nq = manipuland->num_positions;
  VectorXd q_old = x_manipuland.block(0, 0, manipuland->num_positions, 1);
  manipuland_kinematics_cache.initialize(q_old);
  manipuland->doKinematics(manipuland_kinematics_cache);
  double now;

  // set up a quadratic program:
  // 0.5 * x.' Q x + f.' x
  // and since we're unconstrained then solve as linear system
  // Qx = -f

  VectorXd f(nq);
  f.setZero();
  MatrixXd Q(nq, nq);
  Q.setZero();
  double K = 0.;

  double icp_var = 0.05; // m
  double joint_known_fb_var = 0.1; // m
  double joint_known_encoder_var = 0.001; // radian
  double joint_limit_var = 0.01; // one-sided, radians
  double position_constraint_var = 0.1; // one-sided, radians

  double dynamics_floating_base_var = 0.0001; // m per frame
  double dynamics_other_var = 0.1; // rad per frame

  double free_space_var = 0.1;

  double ICP_WEIGHT = 1 / (2. * icp_var * icp_var);
  double FREE_SPACE_WEIGHT = 1 / (2. * free_space_var * free_space_var);
  double JOINT_LIMIT_WEIGHT = 1 / (2. * joint_limit_var * joint_limit_var);
  double POSITION_CONSTRAINT_WEIGHT = 1 / (2. * position_constraint_var * position_constraint_var);
  double JOINT_KNOWN_FLOATING_BASE_WEIGHT = 1 / (2. * joint_known_fb_var * joint_known_fb_var);
  double JOINT_KNOWN_ENCODER_WEIGHT = 1 / (2. * joint_known_encoder_var * joint_known_encoder_var);
  double DYNAMICS_FLOATING_BASE_WEIGHT = 1 / (2. * dynamics_floating_base_var * dynamics_floating_base_var);
  double DYNAMICS_OTHER_WEIGHT = 1 / (2. * dynamics_other_var * dynamics_other_var);

  double MAX_CONSIDERED_ICP_DISTANCE = 0.075;
  double MIN_CONSIDERED_JOINT_DISTANCE = 0.03;

  /***********************************************
                Articulated ICP 
    *********************************************/
  if (ICP_WEIGHT > 0){
    now = getUnixTime();

    VectorXd phi(points.cols());
    Matrix3Xd normal(3, points.cols()), x(3, points.cols()), body_x(3, points.cols());
    std::vector<int> body_idx(points.cols());
    // project all cloud points onto the surface of the object positions
    // via the last state estimate
    double now1 = getUnixTime();
    manipuland->collisionDetectFromPoints(manipuland_kinematics_cache, points,
                         phi, normal, x, body_x, body_idx, false);
    //printf("SDF took %f\n", getUnixTime()-now1);

    // for every unique body points have returned onto...
    std::vector<int> num_points_on_body(manipuland->bodies.size(), 0);

    for (int i=0; i < body_idx.size(); i++)
      num_points_on_body[body_idx[i]] += 1;

    // for every body...
    for (int i=0; i < manipuland->bodies.size(); i++){
      if (num_points_on_body[i] > 0){
        // collect results from raycast that correspond to this sensor
        Matrix3Xd z(3, num_points_on_body[i]); // points, in world frame, near this body
        Matrix3Xd z_prime(3, num_points_on_body[i]); // same points projected onto surface of body
        Matrix3Xd body_z_prime(3, num_points_on_body[i]); // projected points in body frame
        Matrix3Xd z_norms(3, num_points_on_body[i]); // normals corresponding to these points
        int k = 0;
        for (int j=0; j < body_idx.size(); j++){
          assert(k < body_idx.size());
          if (body_idx[j] == i){
            assert(j < points.cols());
            if (points(0, j) == 0.0){
              cout << "Zero points " << points.block<3, 1>(0, j).transpose() << " slipping in at bdyidx " << body_idx[j] << endl;
            }
            if ((points.block<3, 1>(0, j) - x.block<3, 1>(0, j)).norm() <= MAX_CONSIDERED_ICP_DISTANCE){
              auto joint = dynamic_cast<const RevoluteJoint *>(&manipuland->bodies[body_idx[j]]->getJoint());
              bool too_close_to_joint = false;
              if (joint){
                // axis in body frame:
                const Vector3d n = joint->getRotationAxis();
                auto p = body_x.block<3, 1>(0, j);

                // distance to that axis:
                double np = p.transpose() * n;
                double dist_to_joint_axis = (p - (np*n)).norm();
                if (dist_to_joint_axis <= MIN_CONSIDERED_JOINT_DISTANCE){
                  too_close_to_joint = true;
                }
              }

              if (too_close_to_joint == false){
                z.block<3, 1>(0, k) = points.block<3, 1>(0, j);
                z_prime.block<3, 1>(0, k) = x.block<3, 1>(0, j);
                body_z_prime.block<3, 1>(0, k) = body_x.block<3, 1>(0, j);
                z_norms.block<3, 1>(0, k) = normal.block<3, 1>(0, j);
                k++;
              }
            }
          }
        }

        z.conservativeResize(3, k);
        z_prime.conservativeResize(3, k);
        body_z_prime.conservativeResize(3, k);
        z_norms.conservativeResize(3, k);

        // forwardkin to get our jacobians at the project points on the body
        auto J = manipuland->transformPointsJacobian(manipuland_kinematics_cache, body_z_prime, i, 0, false);

        // apply point-to-plane cost
        // we're minimizing point-to-plane projected distance after moving the body config by delta_q
        // i.e. (z - z_prime_new).' * n
        //   =  (z - (z_prime + J*(q_new - q_old))) .' * n
        //   =  (z - z_prime - J*(q_new - q_old))) .' * n
        // Which, if we penalize quadratically, and expand out, removing constant terms, we get
        // argmin_{qn}[ qn.' * (J.' * n * n.' * J) * qn +
        //              - 2 * (Ks.' * n * n.' * J) ]
        // for Ks = (z - z_prime + Jz*q_old)

        bool POINT_TO_PLANE = false;

        for (int j=0; j < k; j++){
          MatrixXd Ks = z.col(j) - z_prime.col(j) + J.block(3*j, 0, 3, nq)*q_old;
          if (POINT_TO_PLANE){
            //cout << z_norms.col(j).transpose() << endl;
            //cout << "Together: " << (z_norms.col(j) * z_norms.col(j).transpose()) << endl;
            f -= ICP_WEIGHT*(2. * Ks.transpose() * (z_norms.col(j) * z_norms.col(j).transpose()) * J.block(3*j, 0, 3, nq)).transpose();
            Q += ICP_WEIGHT*(2. *  J.block(3*j, 0, 3, nq).transpose() * (z_norms.col(j) * z_norms.col(j).transpose()) * J.block(3*j, 0, 3, nq));
          } else {
            f -= ICP_WEIGHT*(2. * Ks.transpose() * J.block(3*j, 0, 3, nq)).transpose();
            Q += ICP_WEIGHT*(2. *  J.block(3*j, 0, 3, nq).transpose() * J.block(3*j, 0, 3, nq));
          }
          K += ICP_WEIGHT*Ks.squaredNorm();

          if (j % 1 == 0){
            // visualize point correspondences and normals
            if (z(0, j) == 0.0){
              cout << "Got zero z " << z.block<3, 1>(0, j).transpose() << " at z prime " << z_prime.block<3, 1>(0, j).transpose() << endl;
            }
            double dist_normalized = fmin(MAX_CONSIDERED_ICP_DISTANCE, (z.col(j) - z_prime.col(j)).norm()) / MAX_CONSIDERED_ICP_DISTANCE;
   
            bot_lcmgl_begin(lcmgl_icp_, LCMGL_LINES);
            bot_lcmgl_color3f(lcmgl_icp_, dist_normalized*dist_normalized, 0, (1.0-dist_normalized)*(1.0-dist_normalized));
            bot_lcmgl_line_width(lcmgl_icp_, 2.0f);
            bot_lcmgl_vertex3f(lcmgl_icp_, z(0, j), z(1, j), z(2, j));
            bot_lcmgl_vertex3f(lcmgl_icp_, z_prime(0, j), z_prime(1, j), z_prime(2, j));
            bot_lcmgl_end(lcmgl_icp_);  

  /*
            bot_lcmgl_line_width(lcmgl_icp_, 1.0f);
            bot_lcmgl_color3f(lcmgl_icp_, 1.0, 0.0, 1.0);
            bot_lcmgl_begin(lcmgl_icp_, LCMGL_LINES);
            bot_lcmgl_vertex3f(lcmgl_icp_, z_prime(0, j)+z_norms(0, j)*0.01, z_prime(1, j)+z_norms(1, j)*0.01, z_prime(2, j)+z_norms(2, j)*0.01);
            bot_lcmgl_vertex3f(lcmgl_icp_, z_prime(0, j), z_prime(1, j), z_prime(2, j));
            bot_lcmgl_end(lcmgl_icp_);  
  */
          }
        }
      }
    }
    bot_lcmgl_switch_buffer(lcmgl_icp_);  

    //printf("Spend %f in Articulated ICP constraints.\n", getUnixTime() - now);
  }

  /***********************************************
                FREE SPACE CONSTRAINT
    *********************************************/
  if (FREE_SPACE_WEIGHT > 0){
    now = getUnixTime();

    // calculate SDFs in the image plane (not voxel grid like DART... too expensive
    // since we're not on a GPU yet)

    // perform raycast to generate "expected" observation
    // (borrowing code from Matthew Woehlke's pull request for the moment here)
    VectorXd distances(raycast_endpoints.cols());
    Vector3d origin = kinect2world*Vector3d::Zero();
    Matrix3Xd origins(3, raycast_endpoints.cols());
    Matrix3Xd normals(3, raycast_endpoints.cols());
    std::vector<int> body_idx(raycast_endpoints.cols());
    for (int i=0; i < raycast_endpoints.cols(); i++)
      origins.block<3, 1>(0, i) = origin;

    Matrix3Xd raycast_endpoints_world = kinect2world*raycast_endpoints;
    double before_raycast = getUnixTime();
    manipuland->collisionRaycast(manipuland_kinematics_cache,origins,raycast_endpoints_world,distances,normals,body_idx);
    //printf("Raycast took %f\n", getUnixTime() - before_raycast);


    // fix the raycast distances to behave like the kinect distances:
    // i.e. the distance is the z-distance of the intersection point in the camera origin frame
    // and also initialize observation SDF to Inf where the measurement return is in front of the real return
    // and 0 otherwise
    Eigen::MatrixXd observation_sdf_input = MatrixXd::Constant(num_pixel_rows, num_pixel_cols, 0.0);
    for (int i=0; i<num_pixel_rows; i++) {
      for (int j=0; j<num_pixel_cols; j++) {
        // fix distance
        long int thisind = i*num_pixel_cols+j;
        // this could be done with trig instead by I think this is just as efficient?
        distances(thisind) = distances(thisind)*raycast_endpoints(2, thisind)/MAX_SCAN_DIST;
        if (i < depth_image.rows() && j < depth_image.cols() && 
          distances(thisind) > 0. && 
          distances(thisind) < depth_image(i, j)){
          observation_sdf_input(i, j) = INF;
        }
  //      int thisind = i*num_pixel_cols+j;
  //      if (j % 20 == 0 && distances(thisind) > 0.){
  //        Vector3d endpt = kinect2world*(raycast_endpoints.col(thisind)*distances(thisind)/MAX_SCAN_DIST.);
  //        Vector3d endpt2 = kinect2world*Vector3d( (((double) j)- kcal->intrinsics_depth.cx)*depth_image(i, j) / kcal->intrinsics_rgb.fx,
  //                        (((double) i)- kcal->intrinsics_depth.cy)*depth_image(i, j) / kcal->intrinsics_rgb.fx,
  //                        depth_image(i, j));
  //
  //        bot_lcmgl_begin(lcmgl_measurement_model_, LCMGL_LINES);
  //        bot_lcmgl_color3f(lcmgl_measurement_model_, 0, 1, 0);  
  //        bot_lcmgl_vertex3f(lcmgl_measurement_model_, endpt(0), endpt(1), endpt(2));
  //       //bot_lcmgl_color3f(lcmgl_measurement_model_, 1, 0, 0);  
  //        bot_lcmgl_vertex3f(lcmgl_measurement_model_, endpt2(0), endpt2(1), endpt2(2));
  //        bot_lcmgl_end(lcmgl_measurement_model_);  
  //      }

      }
    }
    MatrixXd observation_sdf;
    MatrixXi mapping_row;
    MatrixXi mapping_col;

    df_2d(observation_sdf_input, observation_sdf, mapping_row, mapping_col);
    for (size_t i=0; i<num_pixel_rows; i++) {
      for (size_t j=0; j<num_pixel_cols; j++) {
        observation_sdf(i, j) = sqrtf(observation_sdf(i, j));
      }
    }

    cv::Mat image;
    cv::Mat image_bg;
    eigen2cv(observation_sdf, image);
    eigen2cv(depth_image, image_bg);
    double min, max;
    cv::minMaxIdx(image, &min, &max);
    if (max > 0)
      image = image / max;
    cv::minMaxIdx(image_bg, &min, &max);
    if (max > 0)
      image_bg = image_bg / max;
    cv::Mat image_disp;
    cv::addWeighted(image, 1.0, image_bg, 0.0, 0.0, image_disp);
    cv::resize(image_disp, image_disp, cv::Size(640, 480));
    cv::imshow("IRB140EstimatorDebug", image_disp);

    // calculate projection direction to try to resolve this.
    // following Ganapathi / Thrun 2010, we'll do this by balancing
    // projection in two directions: perpendicular to raycast, and
    // then along it.

    double constant = 1.0f / kcal->intrinsics_rgb.fx ;
    // for every unique body points have returned onto...
    std::vector<int> num_points_on_body(manipuland->bodies.size(), 0);
    for (int bdy_i=0; bdy_i < body_idx.size(); bdy_i++){
      if (body_idx[bdy_i] >= 0)
        num_points_on_body[body_idx[bdy_i]] += 1;
    }

    // for every body...
    for (int bdy_i=0; bdy_i < manipuland->bodies.size(); bdy_i++){
      // assemble correction vectors and points for this body
      if (num_points_on_body[bdy_i] > 0){
        int k = 0;

        // collect simulated depth points, and the corrected points 
        // based on depth map laterally and longitudinally
        Matrix3Xd z(3, num_points_on_body[bdy_i]);
        Matrix3Xd z_corrected_depth = MatrixXd::Constant(3, num_points_on_body[bdy_i], 0.0);
        Matrix3Xd z_corrected_lateral = MatrixXd::Constant(3, num_points_on_body[bdy_i], 0.0);

        enum DepthCorrections { DC_NONE, DC_LATERAL, DC_DEPTH };
        std::vector<DepthCorrections> depth_correction(num_points_on_body[bdy_i], DC_NONE);

        for (int i=0; i<num_pixel_rows; i++) {
          for (int j=0; j<num_pixel_cols; j++) {
            long int thisind = i*num_pixel_cols + j;
            if (body_idx[thisind] == bdy_i){
              // project depth into world:
              Vector3d endpt = origin + distances(thisind) * ((raycast_endpoints_world.block<3, 1>(0, thisind) - origin)/raycast_endpoints(2, thisind));

              // Lateral correction
              Vector3d camera_correction_vector;
              if (observation_sdf(thisind) > 0.0 && observation_sdf(thisind) < INF){
                camera_correction_vector(0) = ((double)(j - mapping_col(i, j)))*distances[thisind]*constant;
                camera_correction_vector(1) = ((double)(i - mapping_row(i, j)))*distances[thisind]*constant;
                camera_correction_vector(2) = 0.0;
                
                Vector3d world_correction_vector = kinect2world.rotation()*camera_correction_vector;

                if (camera_correction_vector.norm() < fabs(distances[thisind] - depth_image(i, j))) { 
                  z_corrected_lateral.block<3,1>(0, k) = endpt + world_correction_vector;
                  z.block<3, 1>(0, k) = endpt;
                  depth_correction[k] = DC_LATERAL;
                  k++;

                  if (thisind % 1 == 0){
                    bot_lcmgl_begin(lcmgl_measurement_model_, LCMGL_LINES);
                    bot_lcmgl_line_width(lcmgl_measurement_model_, 5.0f);
                    bot_lcmgl_color3f(lcmgl_measurement_model_, 0, 0, 1);  
                    bot_lcmgl_vertex3f(lcmgl_measurement_model_, endpt(0), endpt(1), endpt(2));
                    bot_lcmgl_vertex3f(lcmgl_measurement_model_, endpt(0)+world_correction_vector(0), endpt(1)+world_correction_vector(1), endpt(2)+world_correction_vector(2));
                    bot_lcmgl_end(lcmgl_measurement_model_);  
                  }

                }
              } 
              // Depth correction term
              if (observation_sdf(thisind) >= INF && fabs(distances[thisind] - depth_image(i, j)) < 0.5 ){
                // simply push back to "correct" depth
                Vector3d corrected_endpt = origin + depth_image(i, j) * ((raycast_endpoints_world.block<3, 1>(0, thisind) - origin) / raycast_endpoints(2, thisind));
                z.block<3, 1>(0, k) = endpt;
                z_corrected_depth.block<3,1>(0, k) = corrected_endpt;
                depth_correction[k] = DC_DEPTH;
                k++;

                if (thisind % 1 == 0){
                  bot_lcmgl_begin(lcmgl_measurement_model_, LCMGL_LINES);
                  bot_lcmgl_line_width(lcmgl_measurement_model_, 5.0f);
                  bot_lcmgl_color3f(lcmgl_measurement_model_, 0, 1, 0);  
                  bot_lcmgl_vertex3f(lcmgl_measurement_model_, endpt(0), endpt(1), endpt(2));
                  bot_lcmgl_vertex3f(lcmgl_measurement_model_, corrected_endpt(0), corrected_endpt(1), corrected_endpt(2));
                  bot_lcmgl_end(lcmgl_measurement_model_);
                }
              }
            }
          }
        }

        z.conservativeResize(3, k);
        z_corrected_depth.conservativeResize(3, k);
        z_corrected_lateral.conservativeResize(3, k);

        // now do an icp step attempting to resolve said constraints
        
        // forwardkin the points in the body frame
        Matrix3Xd z_body = manipuland->transformPoints(manipuland_kinematics_cache, z, 0, bdy_i);
        // forwardkin to get our jacobians at the project points on the body
        auto J = manipuland->transformPointsJacobian(manipuland_kinematics_cache, z_body, bdy_i, 0, false);

        // apply corrections in the big linear solve
        for (int j=0; j < z.cols(); j++){
          MatrixXd Ks(3, 1);
          if (depth_correction[j] == DC_DEPTH){
            Ks = z_corrected_depth.col(j) - z.col(j) + J.block(3*j, 0, 3, nq)*q_old;
          } else if (depth_correction[j] == DC_LATERAL) {
            Ks = z_corrected_lateral.col(j) - z.col(j) + J.block(3*j, 0, 3, nq)*q_old;
          } else {
            continue;
          }
          f -= FREE_SPACE_WEIGHT*(2. * Ks.transpose() * J.block(3*j, 0, 3, nq)).transpose();
          Q += FREE_SPACE_WEIGHT*(2. *  J.block(3*j, 0, 3, nq).transpose() * J.block(3*j, 0, 3, nq));
          K += FREE_SPACE_WEIGHT*Ks.squaredNorm();
        }
      }
    }

    bot_lcmgl_point_size(lcmgl_measurement_model_, 4.0f);
    bot_lcmgl_color3f(lcmgl_measurement_model_, 0, 0, 1);  
    bot_lcmgl_begin(lcmgl_measurement_model_, LCMGL_POINTS);
    for (int i = 0; i < distances.rows(); i++){
      if (i % 1 == 0){
        Vector3d endpt = origin + distances(i) * ((raycast_endpoints_world.block<3, 1>(0, i) - origin) / raycast_endpoints(2, i));
        if (endpt(0) > manip_x_bounds[0] && endpt(0) < manip_x_bounds[1] && 
            endpt(1) > manip_y_bounds[0] && endpt(1) < manip_y_bounds[1] && 
            endpt(2) > manip_z_bounds[0] && endpt(2) < manip_z_bounds[1] &&
            (1 || observation_sdf(i) > 0.0 && observation_sdf(i) < INF)) {
          bot_lcmgl_vertex3f(lcmgl_measurement_model_, endpt(0), endpt(1), endpt(2));
        }
      }
    }
    bot_lcmgl_end(lcmgl_measurement_model_);
    bot_lcmgl_switch_buffer(lcmgl_measurement_model_);  

    //printf("Spend %f in free space constraints.\n", getUnixTime() - now);
  }

  /***********************************************
                DYNAMICS HINTS
    *********************************************/
  if (DYNAMICS_OTHER_WEIGHT > 0 || DYNAMICS_FLOATING_BASE_WEIGHT > 0){
    now = getUnixTime();
    // for now, foh on dynamics
    // min (x - x')^2
    // i.e. min x^2 - 2xx' + x'^2
    for (int i=0; i<6; i++){
      Q(i, i) += DYNAMICS_FLOATING_BASE_WEIGHT*1.0;
      f(i) -= DYNAMICS_FLOATING_BASE_WEIGHT*q_old(i);
      K += DYNAMICS_FLOATING_BASE_WEIGHT*q_old(i)*q_old(i);
    }
    for (int i=6; i<q_old.rows(); i++){
      Q(i, i) += DYNAMICS_OTHER_WEIGHT*1.0;
      f(i) -= DYNAMICS_OTHER_WEIGHT*q_old(i);
      K += DYNAMICS_OTHER_WEIGHT*q_old(i)*q_old(i);
    }
    //printf("Spent %f in joint known weight constraints.\n", getUnixTime() - now);
  }


  /***********************************************
                KNOWN POSITION HINTS
    *********************************************/
  if (JOINT_KNOWN_ENCODER_WEIGHT > 0 || JOINT_KNOWN_FLOATING_BASE_WEIGHT > 0){
    now = getUnixTime();
    // min (x - x')^2
    // i.e. min x^2 - 2xx' + x'^2
    x_manipuland_measured_mutex.lock();
    VectorXd q_measured = x_manipuland_measured.block(0,0,nq,1);
    std::vector<bool> x_manipuland_measured_known_copy = x_manipuland_measured_known;
    x_manipuland_measured_mutex.unlock();

    for (int i=0; i<6; i++){
      if (x_manipuland_measured_known_copy[i]){
        Q(i, i) += JOINT_KNOWN_FLOATING_BASE_WEIGHT*1.0;
        f(i) -= JOINT_KNOWN_FLOATING_BASE_WEIGHT*q_measured(i);
        K += JOINT_KNOWN_FLOATING_BASE_WEIGHT*q_measured(i)*q_measured(i);
      }
    }
    for (int i=6; i<q_old.rows(); i++){
      if (x_manipuland_measured_known_copy[i]){
        Q(i, i) += JOINT_KNOWN_ENCODER_WEIGHT*1.0;
        f(i) -= JOINT_KNOWN_ENCODER_WEIGHT*q_measured(i);
        K += JOINT_KNOWN_ENCODER_WEIGHT*q_measured(i)*q_measured(i);
      }
    }
    //printf("Spent %f in joint known weight constraints.\n", getUnixTime() - now);
  }

  /***********************************************
                JOINT LIMIT CONSTRAINTS
    *********************************************/
  if (JOINT_LIMIT_WEIGHT > 0){
    now = getUnixTime();
    // push negative ones back towards their limits
    // phi_jl(i) = J_jl(i,i)*(x - lim)
    // (back out lim = x - phi_jl(i)/J_jl(i,i)
    // min phi_li^2 if phi_jl < 0, so
    // min (J_jl(i,i)*(x-lim))^2
    // min x^2 - 2 * lim * x + lim^2
    for (int i=0; i<q_old.rows(); i++){
      if (isfinite(manipuland->joint_limit_min[i]) && q_old[i] < manipuland->joint_limit_min[i]){
        Q(i, i) += JOINT_LIMIT_WEIGHT*1.0;
        f(i) -= JOINT_LIMIT_WEIGHT*manipuland->joint_limit_min[i];
        K += JOINT_LIMIT_WEIGHT*manipuland->joint_limit_min[i]*manipuland->joint_limit_min[i];
      }
      if (isfinite(manipuland->joint_limit_max[i]) && q_old[i] > manipuland->joint_limit_max[i]){
        Q(i, i) += JOINT_LIMIT_WEIGHT*1.0;
        f(i) -= JOINT_LIMIT_WEIGHT*manipuland->joint_limit_max[i];
        K += JOINT_LIMIT_WEIGHT*manipuland->joint_limit_max[i]*manipuland->joint_limit_max[i];
      }
    }
    //printf("Spent %f in joint limit constraints.\n", getUnixTime() - now);
  }

  /***********************************************
                POSITION CONSTRAINTS
    *********************************************/
  if (POSITION_CONSTRAINT_WEIGHT > 0){
    now = getUnixTime();

    VectorXd positionConstraints = manipuland->positionConstraints(manipuland_kinematics_cache);
    MatrixXd positionConstraintsJ = manipuland->positionConstraintsJacobian(manipuland_kinematics_cache, true);
    for (int i=0; i < positionConstraints.rows(); i++){
      // push nonzero ones back towards zero
      // phi_pc_new(i) = J_pc(i, :) * (q - q_old) + phi_pc(i)
      // min [ phi_pc_new(i)^2 ]
      // which you can expand out... it's pretty big.      
      MatrixXd Jpc = POSITION_CONSTRAINT_WEIGHT * positionConstraintsJ.block(i,0,1,nq);      
      Q.block(0, 0, nq, nq) += POSITION_CONSTRAINT_WEIGHT * Jpc.transpose() * Jpc;
      f.block(0, 0, nq, 1) += POSITION_CONSTRAINT_WEIGHT * (-1.0 * q_old.transpose() * Jpc.transpose() * Jpc + positionConstraints(i) * Jpc).transpose();      
      K += 0.5 * POSITION_CONSTRAINT_WEIGHT * positionConstraints(i) * positionConstraints(i);      
      K += 0.5 * POSITION_CONSTRAINT_WEIGHT * q_old.transpose() * Jpc.transpose() * Jpc * q_old;      
      K += -1.0 * POSITION_CONSTRAINT_WEIGHT * (positionConstraints(i) * Jpc * q_old)(0);      
    }
    //printf("Spent %f in position constraints.\n", getUnixTime() - now);
  }

  /***********************************************
                       SOLVE
    *********************************************/
  if (K > 0.0){
    //cout << "f: " << f << endl;
    //cout << "Q: " << Q << endl;
    //cout << "K: " << K << endl;
    // Solve the unconstrained QP!
    VectorXd q_new = Q.colPivHouseholderQr().solve(-f);
    //cout << "q_new: " << q_new.transpose() << endl;

    // apply joint lim
    /*
    for (int i=0; i<q_new.rows(); i++){
      if (isfinite(manipuland->joint_limit_min[i]))
        q_new[i] = fmax(manipuland->joint_limit_min[i], q_new[i]);
      if (isfinite(manipuland->joint_limit_max[i]))
        q_new[i] = fmin(manipuland->joint_limit_max[i], q_new[i]);
    }*/
    

    x_manipuland.block(0, 0, nq, 1) = q_new; //x_manipuland.block(0, 0, nq, 1)*0.5 + 0.5*q_new;
  }

}

void IRB140Estimator::setupSubscriptions(const char* state_channelname,
  const char* hand_state_channelname){
  //lcm->subscribe("SCAN", &IRB140EstimatorSystem::handlePointlatest_cloud, this);
  //lcm.subscribe("SCAN", &IRB140Estimator::handlePlanarLidarMsg, this);
  //lcm.subscribe("PRE_SPINDLE_TO_POST_SPINDLE", &IRB140Estimator::handleSpindleFrameMsg, this);
  auto kinect_frame_sub = lcm.subscribe("KINECT_FRAME", &IRB140Estimator::handleKinectFrameMsg, this);
  kinect_frame_sub->setQueueCapacity(1);
  auto state_sub = lcm.subscribe(state_channelname, &IRB140Estimator::handleRobotStateMsg, this);
  state_sub->setQueueCapacity(1);
  auto save_pc_sub = lcm.subscribe("IRB140_ESTIMATOR_SAVE_POINTCLOUD", &IRB140Estimator::handleSavePointcloudMsg, this);
  save_pc_sub->setQueueCapacity(1);
  auto hand_state_sub = lcm.subscribe(hand_state_channelname, &IRB140Estimator::handleLeftHandStateMsg, this);
  hand_state_sub->setQueueCapacity(1);

}

void IRB140Estimator::handleSavePointcloudMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::raw_t* msg){
  string filename(msg->data.begin(), msg->data.end());
  printf("####Received save command on channel %s to file %s\n", chan.c_str(), filename.c_str());

  Matrix3Xd full_cloud;
  latest_cloud_mutex.lock();
  full_cloud = latest_cloud;
  latest_cloud_mutex.unlock();

  // transform into world frame
  Eigen::Isometry3d kinect2tag;
  long long utime = 0;
  this->get_trans_with_utime("KINECT_RGB", "KINECT_TO_APRILTAG", utime, kinect2tag);
  Eigen::Isometry3d world2tag;
  long long utime2 = 0;
  this->get_trans_with_utime("local", "robot_yplus_tag", utime2, world2tag);
  Eigen::Isometry3d kinect2world =  world2tag.inverse() * kinect2tag;
  full_cloud = kinect2world*full_cloud;

  // save points in the manip bounds
  ofstream ofile(filename.c_str(), ofstream::out);  
  // first point is camera point in world frame
  Eigen::Vector3d camera_point = kinect2world*Eigen::Vector3d::Zero();
  ofile << camera_point[0] << ", " << camera_point[1] << ", " << camera_point[2] << endl;

  // rest are points in workspace in world frame
  for (int i=0; i < full_cloud.cols(); i++){
    Vector3d pt = full_cloud.block<3,1>(0, i);
    if (pt[0] > manip_x_bounds[0] && pt[0] < manip_x_bounds[1] && 
        pt[1] > manip_y_bounds[0] && pt[1] < manip_y_bounds[1] && 
        pt[2] > manip_z_bounds[0] && pt[2] < manip_z_bounds[1]){
      ofile << pt[0] << ", " << pt[1] << ", " << pt[2] << endl;
    }
  }
  ofile.close();
}

void IRB140Estimator::handlePlanarLidarMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::planar_lidar_t* msg){
  printf("Received scan on channel %s\n", chan.c_str());
  // transform according 
}

void IRB140Estimator::handleSpindleFrameMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::rigid_transform_t* msg){
  //printf("Received transform on channel %s\n", chan.c_str());
  //cout << msg->trans << "," << msg->quat << endl;
  // todo: transform them all by the lidar frame
}


void IRB140Estimator::handleLeftHandStateMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::joint_state_t* msg){
  //printf("Received hand state on channel  %s\n", chan.c_str());
  x_manipuland_measured_mutex.lock();

  map<string, int> map = manipuland->computePositionNameToIndexMap();
  for (int i=0; i < msg->num_joints; i++){
    auto id = map.end();
    if (i == 0){
      id = map.find("left_finger_1_joint_1");
    } else if (i == 3) {
      id = map.find("left_finger_2_joint_1");
    } else if (i == 6) {
      id = map.find("left_finger_middle_joint_1");
    }

    if (id != map.end()){
      x_manipuland_measured(id->second) = msg->joint_position[i];
      x_manipuland_measured_known[id->second] = true;
    }
  }

  x_manipuland_measured_mutex.unlock();

}

void IRB140Estimator::handleRobotStateMsg(const lcm::ReceiveBuffer* rbuf,
                         const std::string& chan,
                         const bot_core::robot_state_t* msg){
  //printf("Received robot state on channel  %s\n", chan.c_str());
  x_manipuland_measured_mutex.lock();

  if (transcribe_published_floating_base){
    x_manipuland_measured(0) = msg->pose.translation.x;
    x_manipuland_measured(1) = msg->pose.translation.y;
    x_manipuland_measured(2) = msg->pose.translation.z;

    auto quat = Quaterniond(msg->pose.rotation.w, msg->pose.rotation.x, msg->pose.rotation.y, msg->pose.rotation.z);
    x_manipuland_measured.block<3, 1>(3, 0) = quat.toRotationMatrix().eulerAngles(2, 1, 0);
  }
  for (int i=0; i < 6; i++)
    x_manipuland_measured_known[i] = true;

  map<string, int> map = manipuland->computePositionNameToIndexMap();
  for (int i=0; i < msg->num_joints; i++){
    auto id = map.find(msg->joint_name[i]);
    if (id != map.end()){
      x_manipuland_measured(id->second) = msg->joint_position[i];
      x_manipuland_measured_known[id->second] = true;
    }
  }

  x_manipuland_measured_mutex.unlock();

}


void IRB140Estimator::handleKinectFrameMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const kinect::frame_msg_t* msg){
  //printf("Received kinect frame on channel %s\n", chan.c_str());

  // only dealing with depth. Copied from ddKinectLCM... shouldn't 
  // this be in the Kinect driver or something?

  latest_cloud_mutex.lock();

  std::vector<uint16_t> depth_data;

  // 1.2.1 De-compress if necessary:
  if(msg->depth.compression != msg->depth.COMPRESSION_NONE) {
    // ugh random C code
    uint8_t * uncompress_buffer = (uint8_t*) malloc(msg->depth.uncompressed_size);
    unsigned long dlen = msg->depth.uncompressed_size;
    int status = uncompress(uncompress_buffer, &dlen, 
        msg->depth.depth_data.data(), msg->depth.depth_data_nbytes);
    if(status != Z_OK) {
      printf("Problem in uncompression.\n");
      free(uncompress_buffer);
      latest_cloud_mutex.unlock();
      return;
    }
    for (int i=0; i<msg->depth.uncompressed_size/2; i++)
      depth_data.push_back( ((uint16_t)uncompress_buffer[2*i])+ (((uint16_t)uncompress_buffer[2*i+1])<<8) );
    free(uncompress_buffer);

  }else{
    for (int i=0; i<msg->depth.depth_data.size()/2; i++)
      depth_data.push_back(  ((uint16_t)msg->depth.depth_data[2*i])+ (((uint16_t)msg->depth.depth_data[2*i+1])<<8) );
  }

  if(msg->depth.depth_data_format == msg->depth.DEPTH_MM  ){ 
    /////////////////////////////////////////////////////////////////////
    // Openni Data
    // 1.2.2 unpack raw byte data into float values in mm

    // NB: no depth return is given 0 range - and becomes 0,0,0 here
    if (latest_depth_image.cols() != input_num_pixel_cols || latest_depth_image.rows() != input_num_pixel_rows)
      latest_depth_image.resize(input_num_pixel_rows, input_num_pixel_cols);
    if (latest_cloud.cols() != input_num_pixel_cols*input_num_pixel_rows)
      latest_cloud.resize(3, input_num_pixel_cols*input_num_pixel_rows);

    latest_depth_image.setZero();
    for(long int v=0; v<input_num_pixel_rows; v++) { // t2b self->height 480
      for(long int u=0; u<input_num_pixel_cols; u++ ) {  //l2r self->width 640
        // not dealing with color yet

        double constant = 1.0f / kcal->intrinsics_rgb.fx ;
        double disparity_d = depth_data[v*msg->depth.width+u]  / 1000.; // convert to m

        long int ind = v*input_num_pixel_cols + u;
        latest_cloud(0, ind) = (((double) u)- kcal->intrinsics_depth.cx)*disparity_d*constant; //x right+
        latest_cloud(1, ind) = (((double) v)- kcal->intrinsics_depth.cy)*disparity_d*constant; //y down+
        latest_cloud(2, ind) = disparity_d;  //z forward+
        latest_depth_image(v, u) = disparity_d;
      }
    }
  } else {
    printf("Can't unpack different Kinect data format yet.\n");
  }
  latest_cloud_mutex.unlock();
}