#undef NDEBUG
#include "KinectFrameCost.hpp"

#include <assert.h> 
#include <fstream>
#include "common.hpp"
#include "drake/util/convexHull.h"
#include "zlib.h"
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

KinectFrameCost::KinectFrameCost(std::shared_ptr<RigidBodyTree> robot_, std::shared_ptr<lcm::LCM> lcm_, YAML::Node config) :
    robot(robot_),
    robot_kinematics_cache(robot->bodies),
    lcm(lcm_),
    nq(robot->number_of_positions())
{
  if (config["icp_var"])
    icp_var = config["icp_var"].as<double>();
  if (config["free_space_var"])
    free_space_var = config["free_space_var"].as<double>();
  if (config["max_considered_icp_distance"])
    max_considered_icp_distance = config["max_considered_icp_distance"].as<double>();
  if (config["min_considered_joint_distance"])
    min_considered_joint_distance = config["min_considered_joint_distance"].as<double>();
  if (config["timeout_time"])
    timeout_time = config["timeout_time"].as<double>();
  if (config["max_scan_dist"])
    max_scan_dist = config["max_scan_dist"].as<double>();
  if (config["verbose"])
    verbose = config["verbose"].as<bool>();
  if (config["verbose_lcmgl"])
    verbose_lcmgl = config["verbose_lcmgl"].as<bool>();
  if (config["world_frame"])
    world_frame = config["world_frame"].as<bool>();

  if (config["kinect2world"]){
    have_hardcoded_kinect2robot = true;
    hardcoded_kinect2robot.setIdentity();
    vector<double> hardcoded_tf = config["kinect2world"].as<vector<double>>();
    hardcoded_kinect2robot.matrix().block<3, 1>(0,3) = Vector3d(hardcoded_tf[0], hardcoded_tf[1], hardcoded_tf[2]);
    hardcoded_kinect2robot.matrix().block<3, 3>(0,0) = Quaterniond(hardcoded_tf[3], hardcoded_tf[4], hardcoded_tf[5], hardcoded_tf[6]).toRotationMatrix();
  }

  if (config["bounds"]){
    BoundingBox bounds;
    bounds.xMin = config["bounds"]["xMin"].as<double>();
    bounds.xMax = config["bounds"]["xMax"].as<double>();
    bounds.yMin = config["bounds"]["yMin"].as<double>();
    bounds.yMax = config["bounds"]["yMax"].as<double>();
    bounds.zMin = config["bounds"]["zMin"].as<double>();
    bounds.zMax = config["bounds"]["zMax"].as<double>();
    setBounds(bounds);
  }

  const char * filename = NULL;
  if (config["filename"])
    filename = config["filename"].as<string>().c_str();
  this->initBotConfig(filename);


  lcmgl_lidar_= bot_lcmgl_init(lcm->getUnderlyingLCM(), "trimmed_lidar");
  lcmgl_icp_= bot_lcmgl_init(lcm->getUnderlyingLCM(), "icp_p2pl");
  lcmgl_measurement_model_ = bot_lcmgl_init(lcm->getUnderlyingLCM(), "meas_model");

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

  num_pixel_cols = (int) floor( ((double)input_num_pixel_cols) / downsample_amount);
  num_pixel_rows = (int) floor( ((double)input_num_pixel_rows) / downsample_amount);

  latest_depth_image.resize(input_num_pixel_rows, input_num_pixel_cols);
  raycast_endpoints.resize(3,num_pixel_rows*num_pixel_cols);

  cv::namedWindow( "KinectFrameCostDebug", cv::WINDOW_AUTOSIZE );
  cv::startWindowThread();

  if (!have_hardcoded_kinect2robot){
    auto camera_offset_sub = lcm->subscribe("GT_CAMERA_OFFSET", &KinectFrameCost::handleCameraOffsetMsg, this);
    camera_offset_sub->setQueueCapacity(1);
  }

  auto kinect_frame_sub = lcm->subscribe("KINECT_FRAME", &KinectFrameCost::handleKinectFrameMsg, this);
  kinect_frame_sub->setQueueCapacity(1);

  auto save_pc_sub = lcm->subscribe("IRB140_ESTIMATOR_SAVE_POINTCLOUD", &KinectFrameCost::handleSavePointcloudMsg, this);
  save_pc_sub->setQueueCapacity(1);

  lastReceivedTime = getUnixTime() - timeout_time*2.;
  last_got_kinect_frame = getUnixTime() - timeout_time*2.;
}

void KinectFrameCost::initBotConfig(const char* filename)
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

int KinectFrameCost::get_trans_with_utime(std::string from_frame, std::string to_frame,
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
bool KinectFrameCost::constructCost(ManipulationTracker * tracker, const Eigen::Matrix<double, Eigen::Dynamic, 1> x_old, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Q, Eigen::Matrix<double, Eigen::Dynamic, 1>& f, double& K)
{
  double now = getUnixTime();

  if (now - lastReceivedTime > timeout_time || (!have_hardcoded_kinect2robot && (world_frame && now - last_got_kinect_frame > timeout_time))){
    if (verbose)
      printf("KinectFrameCost: constructed but timed out\n");
    return false;
  }
  else {
    VectorXd q_old = x_old.block(0, 0, robot->number_of_positions(), 1);
    robot_kinematics_cache.initialize(q_old);
    robot->doKinematics(robot_kinematics_cache);

    // pull points from buffer
    Eigen::Matrix3Xd full_cloud;
    Eigen::MatrixXd full_depth_image;
    latest_cloud_mutex.lock();
    full_cloud = latest_cloud;
    full_depth_image.resize(latest_depth_image.rows(), latest_depth_image.cols());
    full_depth_image= latest_depth_image;
    latest_cloud_mutex.unlock();
    
    // transform into world frame?
    Eigen::Isometry3d kinect2world;
    kinect2world.setIdentity();
    if (world_frame){
      long long utime = 0;
      Eigen::Isometry3d robot2world;
      this->get_trans_with_utime("robot_base", "local", utime, robot2world);
      long long utime2 = 0;
      if (have_hardcoded_kinect2robot){
        kinect2world = robot2world * hardcoded_kinect2robot.inverse();
      } else {
        camera_offset_mutex.lock();
        kinect2world =  robot2world * kinect2robot.inverse();
        camera_offset_mutex.unlock();
      }
    }

    //kinect2world.setIdentity();
    full_cloud = kinect2world * full_cloud;

    // do randomized downsampling, populating data stores to be used by the ICP
    Matrix3Xd points(3, full_cloud.cols()); int i=0;
    Eigen::MatrixXd depth_image; depth_image.resize(num_pixel_rows, num_pixel_cols);
    double constant = 1.0f / kcal->intrinsics_rgb.fx ;
    if (verbose_lcmgl)
      bot_lcmgl_begin(lcmgl_lidar_, LCMGL_POINTS);
    if (full_cloud.cols() > 0){
      if (full_cloud.cols() != input_num_pixel_cols*input_num_pixel_rows){
        printf("KinectFramecost: WARNING: SOMEHOW FULL CLOUD HAS WRONG NUMBER OF ENTRIES.\n");
      }
      for (size_t v=0; v<num_pixel_rows; v++) {
        for (size_t u=0; u<num_pixel_cols; u++) {
          int full_v = min((int)floor(((double)v)*downsample_amount) + rand()%(int)downsample_amount, input_num_pixel_rows-1);
          int full_u = min((int)floor(((double)u)*downsample_amount) + rand()%(int)downsample_amount, input_num_pixel_cols-1);

          // cut down to just point cloud in our manipulation space
          Eigen::Vector3d pt = full_cloud.block<3, 1>(0, full_v*input_num_pixel_cols + full_u);
          if (full_depth_image(full_v, full_u) > 0. &&
              pt[0] > pointcloud_bounds.xMin && pt[0] < pointcloud_bounds.xMax && 
              pt[1] > pointcloud_bounds.yMin && pt[1] < pointcloud_bounds.yMax && 
              pt[2] > pointcloud_bounds.zMin && pt[2] < pointcloud_bounds.zMax){
            points.block<3, 1>(0, i) = pt;
            i++;

            if (verbose_lcmgl && (v*num_pixel_cols + u) % 5 == 0){
              bot_lcmgl_color3f(lcmgl_lidar_, 0.5, 1.0, 0.5);
              bot_lcmgl_vertex3f(lcmgl_lidar_, pt[0], pt[1], pt[2]);
            }
          } else {
            if (verbose_lcmgl && (v*num_pixel_cols + u) % 5 == 0){
              bot_lcmgl_color3f(lcmgl_lidar_, 1.0, 0.5, 0.5);
              bot_lcmgl_vertex3f(lcmgl_lidar_, pt[0], pt[1], pt[2]);
            }
          }

          // populate depth image using our random sample
          depth_image(v, u) = full_depth_image(full_v, full_u); 

          // populate raycast endpoints using our random sample
          raycast_endpoints.col(num_pixel_cols*v + u) = Vector3d(
            (((double) full_u)- kcal->intrinsics_depth.cx)*1.0*constant,
            (((double) full_v)- kcal->intrinsics_depth.cy)*1.0*constant, 
            1.0); // simulate the depth sensor;
          raycast_endpoints.col(num_pixel_cols*v + u) *= max_scan_dist/(raycast_endpoints.col(num_pixel_cols*v + u).norm());
        }
      }
    } else {
      printf("KinectFramecost: No points to work with\n");
    }
    if (verbose_lcmgl){
      bot_lcmgl_end(lcmgl_lidar_);
      bot_lcmgl_switch_buffer(lcmgl_lidar_);
    }
    // conservativeResize keeps old coefficients
    // (regular resize would clear them)
    points.conservativeResize(3, i);


    /***********************************************
                  Articulated ICP 
      *********************************************/
    if (!std::isinf(icp_var)){
      double ICP_WEIGHT = 1. / (2. * icp_var * icp_var);
      now = getUnixTime();

      VectorXd phi(points.cols());
      Matrix3Xd normal(3, points.cols()), x(3, points.cols()), body_x(3, points.cols());
      std::vector<int> body_idx(points.cols());
      // project all cloud points onto the surface of the object positions
      // via the last state estimate
      double now1 = getUnixTime();
      robot->collisionDetectFromPoints(robot_kinematics_cache, points,
                           phi, normal, x, body_x, body_idx, false);
      if (verbose)
        printf("SDF took %f\n", getUnixTime()-now1);

      // for every unique body points have returned onto...
      std::vector<int> num_points_on_body(robot->bodies.size(), 0);

      for (int i=0; i < body_idx.size(); i++)
        num_points_on_body[body_idx[i]] += 1;

      // for every body...
      for (int i=0; i < robot->bodies.size(); i++){
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
              if ((points.block<3, 1>(0, j) - x.block<3, 1>(0, j)).norm() <= max_considered_icp_distance){
                auto joint = dynamic_cast<const RevoluteJoint *>(&robot->bodies[body_idx[j]]->getJoint());
                bool too_close_to_joint = false;
                if (joint){
                  // axis in body frame:
                  const Vector3d n = joint->getRotationAxis();
                  auto p = body_x.block<3, 1>(0, j);

                  // distance to that axis:
                  double np = p.transpose() * n;
                  double dist_to_joint_axis = (p - (np*n)).norm();
                  if (dist_to_joint_axis <= min_considered_joint_distance){
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
          auto J = robot->transformPointsJacobian(robot_kinematics_cache, body_z_prime, i, 0, false);

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
              f.block(0, 0, nq, 1) -= ICP_WEIGHT*(2. * Ks.transpose() * (z_norms.col(j) * z_norms.col(j).transpose()) * J.block(3*j, 0, 3, nq)).transpose();
              Q.block(0, 0, nq, nq) += ICP_WEIGHT*(2. *  J.block(3*j, 0, 3, nq).transpose() * (z_norms.col(j) * z_norms.col(j).transpose()) * J.block(3*j, 0, 3, nq));
            } else {
              f.block(0, 0, nq, 1) -= ICP_WEIGHT*(2. * Ks.transpose() * J.block(3*j, 0, 3, nq)).transpose();
              Q.block(0, 0, nq, nq) += ICP_WEIGHT*(2. *  J.block(3*j, 0, 3, nq).transpose() * J.block(3*j, 0, 3, nq));
            }
            K += ICP_WEIGHT*Ks.squaredNorm();

            if (j % 1 == 0){
              // visualize point correspondences and normals
              if (z(0, j) == 0.0){
                cout << "Got zero z " << z.block<3, 1>(0, j).transpose() << " at z prime " << z_prime.block<3, 1>(0, j).transpose() << endl;
              }
              double dist_normalized = fmin(max_considered_icp_distance, (z.col(j) - z_prime.col(j)).norm()) / max_considered_icp_distance;
     
              if (verbose_lcmgl){
                bot_lcmgl_begin(lcmgl_icp_, LCMGL_LINES);
                bot_lcmgl_color3f(lcmgl_icp_, dist_normalized*dist_normalized, 0, (1.0-dist_normalized)*(1.0-dist_normalized));
                bot_lcmgl_line_width(lcmgl_icp_, 2.0f);
                bot_lcmgl_vertex3f(lcmgl_icp_, z(0, j), z(1, j), z(2, j));
                bot_lcmgl_vertex3f(lcmgl_icp_, z_prime(0, j), z_prime(1, j), z_prime(2, j));
                bot_lcmgl_end(lcmgl_icp_);  
              }
            }
          }
        }
      }
      if (verbose_lcmgl){
        bot_lcmgl_switch_buffer(lcmgl_icp_);  
      }

      if (verbose)
        printf("Spend %f in Articulated ICP constraints.\n", getUnixTime() - now);
    }

    /***********************************************
                  FREE SPACE CONSTRAINT
      *********************************************/
    if (!std::isinf(free_space_var)){
      double FREE_SPACE_WEIGHT = 1. / (2. * free_space_var * free_space_var);

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
      robot->collisionRaycast(robot_kinematics_cache,origins,raycast_endpoints_world,distances,normals,body_idx);
      if (verbose)
        printf("Raycast took %f\n", getUnixTime() - before_raycast);


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
          distances(thisind) = distances(thisind)*raycast_endpoints(2, thisind)/max_scan_dist;
          if (i < depth_image.rows() && j < depth_image.cols() && 
            distances(thisind) > 0. && 
            distances(thisind) < depth_image(i, j)){
            observation_sdf_input(i, j) = INF;
          }
    //      int thisind = i*num_pixel_cols+j;
    //      if (j % 20 == 0 && distances(thisind) > 0.){
    //        Vector3d endpt = kinect2world*(raycast_endpoints.col(thisind)*distances(thisind)/max_scan_dist.);
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
      cv::imshow("KinectFrameCostDebug", image_disp);

      // calculate projection direction to try to resolve this.
      // following Ganapathi / Thrun 2010, we'll do this by balancing
      // projection in two directions: perpendicular to raycast, and
      // then along it.

      double constant = 1.0f / kcal->intrinsics_rgb.fx ;
      // for every unique body points have returned onto...
      std::vector<int> num_points_on_body(robot->bodies.size(), 0);
      for (int bdy_i=0; bdy_i < body_idx.size(); bdy_i++){
        if (body_idx[bdy_i] >= 0)
          num_points_on_body[body_idx[bdy_i]] += 1;
      }

      // for every body...
      for (int bdy_i=0; bdy_i < robot->bodies.size(); bdy_i++){
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

                    if (verbose_lcmgl && thisind % 1 == 0){
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

                  if (verbose_lcmgl && thisind % 1 == 0){
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
          Matrix3Xd z_body = robot->transformPoints(robot_kinematics_cache, z, 0, bdy_i);
          // forwardkin to get our jacobians at the project points on the body
          auto J = robot->transformPointsJacobian(robot_kinematics_cache, z_body, bdy_i, 0, false);

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
            f.block(0, 0, nq, 1) -= FREE_SPACE_WEIGHT*(2. * Ks.transpose() * J.block(3*j, 0, 3, nq)).transpose();
            Q.block(0, 0, nq, nq) += FREE_SPACE_WEIGHT*(2. *  J.block(3*j, 0, 3, nq).transpose() * J.block(3*j, 0, 3, nq));
            K += FREE_SPACE_WEIGHT*Ks.squaredNorm();
          }
        }
      }

      if (verbose_lcmgl){
        bot_lcmgl_point_size(lcmgl_measurement_model_, 4.0f);
        bot_lcmgl_color3f(lcmgl_measurement_model_, 0, 0, 1);  
        bot_lcmgl_begin(lcmgl_measurement_model_, LCMGL_POINTS);
        for (int i = 0; i < distances.rows(); i++){
          if (i % 1 == 0){
            Vector3d endpt = origin + distances(i) * ((raycast_endpoints_world.block<3, 1>(0, i) - origin) / raycast_endpoints(2, i));
            if (endpt(0) > pointcloud_bounds.xMin && endpt(0) < pointcloud_bounds.xMax && 
                endpt(1) > pointcloud_bounds.yMin && endpt(1) < pointcloud_bounds.yMax && 
                endpt(2) > pointcloud_bounds.zMin && endpt(2) < pointcloud_bounds.zMax &&
                (1 || observation_sdf(i) > 0.0 && observation_sdf(i) < INF)) {
              bot_lcmgl_vertex3f(lcmgl_measurement_model_, endpt(0), endpt(1), endpt(2));
            }
          }
        }
        bot_lcmgl_end(lcmgl_measurement_model_);
        bot_lcmgl_switch_buffer(lcmgl_measurement_model_);  
      }
      if (verbose)
        printf("Spend %f in free space constraints.\n", getUnixTime() - now);
    }

    return true;
  }
}


void KinectFrameCost::handleCameraOffsetMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const vicon::body_t* msg){
  camera_offset_mutex.lock();
  Vector3d trans(msg->trans[0], msg->trans[1], msg->trans[2]);
  Quaterniond rot(msg->quat[0], msg->quat[1], msg->quat[2], msg->quat[3]);
  kinect2robot.setIdentity();
  kinect2robot.matrix().block<3, 3>(0,0) = rot.matrix();
  kinect2robot.matrix().block<3, 1>(0,3) = trans;
  camera_offset_mutex.unlock();

  last_got_kinect_frame = getUnixTime();
}


void KinectFrameCost::handleSavePointcloudMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::raw_t* msg){
  string filename(msg->data.begin(), msg->data.end());
  printf("####Received save command on channel %s to file %s\n", chan.c_str(), filename.c_str());

  Matrix3Xd full_cloud;
  MatrixXd color_image ;
  latest_cloud_mutex.lock();
  full_cloud = latest_cloud;
  color_image = latest_color_image;
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

  // rest are points in workspace in world frame, followed by color (r g b)
  for (int i=0; i < full_cloud.cols(); i++){
    Vector3d pt = full_cloud.block<3,1>(0, i);
    Vector3d color = color_image.block<3, 1>(0, i);
    if (    pt[0] > pointcloud_bounds.xMin && pt[0] < pointcloud_bounds.xMax && 
            pt[1] > pointcloud_bounds.yMin && pt[1] < pointcloud_bounds.yMax && 
            pt[2] > pointcloud_bounds.zMin && pt[2] < pointcloud_bounds.zMax){
      ofile << pt[0] << ", " << pt[1] << ", " << pt[2] << ", " << color[0] << ", " << color[1] << ", " << color[2] << endl;
    }
  }
  ofile.close();
}

void KinectFrameCost::handleKinectFrameMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const kinect::frame_msg_t* msg){
  if (verbose)
    printf("Received kinect frame on channel %s\n", chan.c_str());

  // only dealing with depth. Copied from ddKinectLCM... shouldn't 
  // this be in the Kinect driver or something?

  latest_cloud_mutex.lock();

  // grab the color image to colorize lidar
  bool success = false;
  cv::Mat decodedImage;
  if (msg->image.image_data_format == kinect::image_msg_t::VIDEO_RGB_JPEG){
   decodedImage = cv::imdecode(msg->image.image_data, 0);
   success = (decodedImage.rows > 0);
  } else {
   printf("Got a Kinect color image in a format I don't understand: %d\n", msg->image.image_data_format);
  }
  if (success){
   if (latest_color_image.cols() != (msg->image.height * msg->image.width)){
     latest_color_image.resize(3, msg->image.height * msg->image.width);
   }
   for(long int v=0; v<msg->image.height; v++) { // t2b self->height 480
     for(long int u=0; u<msg->image.width; u++ ) {  //l2r self->width 640
      long int ind = v*msg->image.width + u;
      latest_color_image(0, ind) = ((float) decodedImage.at<Vec3b>(v, u).val[0]) / 255.0;
      latest_color_image(1, ind) = ((float) decodedImage.at<Vec3b>(v, u).val[1]) / 255.0;
      latest_color_image(2, ind) = ((float) decodedImage.at<Vec3b>(v, u).val[2]) / 255.0;
     }
   }
  }

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

  lastReceivedTime = getUnixTime();
}