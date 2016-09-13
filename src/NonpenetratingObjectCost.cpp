#undef NDEBUG
#include "NonpenetratingObjectCost.hpp"

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

  //lcmgl_lidar_= bot_lcmgl_init(lcm->getUnderlyingLCM(), "trimmed_lidar");
  //lcmgl_icp_= bot_lcmgl_init(lcm->getUnderlyingLCM(), "icp_p2pl");
  //lcmgl_measurement_model_ = bot_lcmgl_init(lcm->getUnderlyingLCM(), "meas_model");

  cv::namedWindow( "NonpenetratingObjectCostDebug", cv::WINDOW_AUTOSIZE );
  cv::startWindowThread();

  //auto kinect_frame_sub = lcm->subscribe("KINECT_FRAME", &NonpenetratingObjectCost::handleKinectFrameMsg, this);
  //kinect_frame_sub->setQueueCapacity(1);

  //uto save_pc_sub = lcm->subscribe("IRB140_ESTIMATOR_SAVE_POINTCLOUD", &NonpenetratingObjectCost::handleSavePointcloudMsg, this);
  //save_pc_sub->setQueueCapacity(1);

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
    return false;
  }

  // TODO: LOOOOOOOTS TO DO HERE

  // First, convert x_old into corresponding q values for robot and robot_object

  // Use raycasting on robot_object to get a smattering of points on object surface

  // Downsample object surface points

  // Loop over points, colliding them with robot() geometry to find nearest-point correspondences

  // Proceed as usual... ICP

  return true;
}
