#undef NDEBUG
#include "GelsightCost.hpp"

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

GelsightCost::GelsightCost(std::shared_ptr<RigidBodyTree> robot_, std::shared_ptr<lcm::LCM> lcm_, YAML::Node config) :
    robot(robot_),
    robot_kinematics_cache(robot->bodies),
    lcm(lcm_),
    nq(robot->num_positions)
{
  if (config["downsample_amount"])
    downsample_amount = config["downsample_amount"].as<double>();
  if (config["gelsight_freespace_var"])
    gelsight_freespace_var = config["gelsight_freespace_var"].as<double>();
  if (config["gelsight_depth_var"])
    gelsight_depth_var = config["gelsight_depth_var"].as<double>();
  if (config["timeout_time"])
    timeout_time = config["timeout_time"].as<double>();
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
    sensor_body_id = robot->findLinkId(config["surface"]["body"].as<string>());
  } else {
    printf("Must define image plane to use a Gelsight cost\n");
    exit(1);
  }

  num_pixel_cols = (int) floor( ((double)input_num_pixel_cols) / downsample_amount);
  num_pixel_rows = (int) floor( ((double)input_num_pixel_rows) / downsample_amount);

  lcmgl_gelsight_ = bot_lcmgl_init(lcm->getUnderlyingLCM(), "gelsight");

  latest_gelsight_image.resize(input_num_pixel_rows, input_num_pixel_cols);

  cv::namedWindow( "GelsightDepth", cv::WINDOW_AUTOSIZE );
  cv::startWindowThread();

  auto gelsight_frame_sub = lcm->subscribe("GELSIGHT_CONTACT", &GelsightCost::handleGelsightFrameMsg, this);
  gelsight_frame_sub->setQueueCapacity(1);

  lastReceivedTime = getUnixTime() - timeout_time*2.;
}

/***********************************************
            KNOWN POSITION HINTS
*********************************************/
bool GelsightCost::constructCost(ManipulationTracker * tracker, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Q, Eigen::Matrix<double, Eigen::Dynamic, 1>& f, double& K)
{
  double now = getUnixTime();
  if (now - lastReceivedTime > timeout_time){
    if (verbose)
      printf("GelsightCost: constructed but timed out\n");
    return false;
  }
  else {
    VectorXd x_old = tracker->output();
    VectorXd q_old = x_old.block(0, 0, robot->num_positions, 1);
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
        if (val >= 0.1){
          contact_points.block<3,1>(0,num_contact_points) = this_point;
          num_contact_points++;
        } else {
          noncontact_points.block<3,1>(0,num_noncontact_points) = this_point;
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
    bot_lcmgl_switch_buffer(lcmgl_gelsight_);  


    // for each point for which we have contact, attract nearby surfaces.

   /* TODO: gotta play tricks with the active collision groups / model.
    for now, probably duplicate the rigidbody tree, one with the gelsightt robot collision gruop added,
    one with it not added.
*/
    if (verbose)
      printf("Spend %f in gelsight constraints.\n", getUnixTime() - now);

    return true;
  }
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
  } else {
   printf("Got a Gelsight image in a format I don't understand: %d\n", msg->pixelformat);
  }
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

  gelsight_frame_mutex.unlock();

  lastReceivedTime = getUnixTime();
}