#include <stdexcept>
#include <iostream>

#include "common/common.hpp"

#include "drake/systems/plants/RigidBodyTree.h"

#include <lcm/lcm-cpp.hpp>

#include "yaml-cpp/yaml.h"
#include "common/common.hpp"
#include "unistd.h"
#include <mutex>

#include "lcmtypes/kinect/frame_msg_t.hpp"
#include "lcmtypes/bot_core/image_t.hpp"
#include <kinect/kinect-utils.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/correspondence.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/transforms.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "zlib.h"

using namespace std;
using namespace Eigen;

typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;
typedef pcl::ReferenceFrame RFType;
typedef pcl::SHOT352 DescriptorType;

pcl::PointCloud<PointType>::Ptr model (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr model_keypoints (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());

//Algorithm params
bool show_keypoints_ (false);
bool show_correspondences_ (false);
bool use_cloud_resolution_ (false);
bool use_hough_ (true);
float model_ss_ (0.01f);
float scene_ss_ (0.03f);
float rf_rad_ (0.015f);
float descr_rad_ (0.02f);
float cg_size_ (0.01f);
float cg_thresh_ (5.0f);
bool verbose_ (true);

// Kinect frame streaming
double lastKinectReceivedTime = -1.0;
std::mutex latest_cloud_mutex;
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> latest_depth_image;
Eigen::Matrix<double, 3, Eigen::Dynamic> latest_color_image;
KinectCalibration* kcal;


double
computeCloudResolution (const pcl::PointCloud<PointType>::ConstPtr &cloud)
{
  double res = 0.0;
  int n_points = 0;
  int nres;
  std::vector<int> indices (2);
  std::vector<float> sqr_distances (2);
  pcl::search::KdTree<PointType> tree;
  tree.setInputCloud (cloud);

  for (size_t i = 0; i < cloud->size (); ++i)
  {
    if (! pcl_isfinite ((*cloud)[i].x))
    {
      continue;
    }
    //Considering the second neighbor since the first is the point itself.
    nres = tree.nearestKSearch (i, 2, indices, sqr_distances);
    if (nres == 2)
    {
      res += sqrt (sqr_distances[1]);
      ++n_points;
    }
  }
  if (n_points != 0)
  {
    res /= n_points;
  }
  return res;
}

class KinectHandlerState{
  public:
    lcm::LCM lcm;
};

void handleKinectFrameMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const kinect::frame_msg_t* msg,
                           KinectHandlerState* state){
  if (verbose_)
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
      latest_color_image(0, ind) = ((float) decodedImage.at<cv::Vec3b>(v, u).val[0]) / 255.0;
      latest_color_image(1, ind) = ((float) decodedImage.at<cv::Vec3b>(v, u).val[1]) / 255.0;
      latest_color_image(2, ind) = ((float) decodedImage.at<cv::Vec3b>(v, u).val[2]) / 255.0;
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

    scene->clear();
    scene_normals->clear();
    scene_descriptors->clear();
    scene_keypoints->clear();

    scene->width = msg->depth.width;
    scene->height = msg->depth.height;
    scene->is_dense = true; // we won't add inf / nan points
    scene->points.resize(scene->width * scene->height);

    // NB: no depth return is given 0 range - and becomes 0,0,0 here
    if (latest_depth_image.cols() != msg->depth.width || latest_depth_image.rows() != msg->depth.height)
      latest_depth_image.resize(msg->depth.height, msg->depth.width);

    latest_depth_image.setZero();
    int j2 = 0;
    for(long int v=0; v<msg->depth.height; v++) { // t2b
      for(long int u=0; u<msg->depth.width; u++ ) {  //l2r 
        // not dealing with color yet

        double constant = 1.0f / kcal->intrinsics_rgb.fx ;
        double disparity_d = depth_data[v*msg->depth.width+u]  / 1000.; // convert to m
        latest_depth_image(v, u) = disparity_d;

        if (!std::isnan(disparity_d) && disparity_d > 0.0){
          scene->points[j2].x = (((double) u)- kcal->intrinsics_depth.cx)*disparity_d*constant;  //x right+
          scene->points[j2].y = (((double) v)- kcal->intrinsics_depth.cy)*disparity_d*constant; //y down+
          scene->points[j2].z = disparity_d; // z forward +
        }
      }
    }
    scene->points.resize(j2);
  } else {
    printf("Can't unpack different Kinect data format yet.\n");
  }
  latest_cloud_mutex.unlock();
  lastKinectReceivedTime = getUnixTime();
}

void setup_kinect_calib(){
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
}

int main(int argc, char** argv) {
  if (argc != 2){
    printf("Use: runObjectDetector <path to yaml config file>\n");
    return -1;
  }

  std::shared_ptr<lcm::LCM> lcm(new lcm::LCM());
  if (!lcm->good()) {
    throw std::runtime_error("LCM is not good");
  }

  setup_kinect_calib();

  char * configFile = argv[1];
  YAML::Node config = YAML::LoadFile(string(configFile)); 

  // 
  // Load model cloud
  //
  if (!config["points"] || !config["normals"]){
    printf("YAML incomplete -- points or normals unspecified\n");
    return -1;
  }

  if (pcl::io::loadPCDFile (config["points"].as<string>(), *model) < 0)
  {
    std::cout << "Error loading model cloud." << std::endl;
    return (-1);
  }
  if (pcl::io::loadPCDFile (config["normals"].as<string>(), *model_normals) < 0)
  {
    std::cout << "Error loading model normals." << std::endl;
    return (-1);
  }
  int num_pts = model->size();
  int num_normals = model_normals->size();
  if (num_pts != num_normals){
    printf("Loaded %d points but %d normals!\n", num_pts, num_normals);
  }
  printf("Loaded %d points and normals for object.\n", num_pts);


  //
  //  Set up resolution invariance
  //
  float resolution = static_cast<float> (computeCloudResolution (model));
  if (use_cloud_resolution_ && resolution != 0.0f)
  {
    model_ss_   *= resolution;
    scene_ss_   *= resolution;
    rf_rad_     *= resolution;
    descr_rad_  *= resolution;
    cg_size_    *= resolution;
  }

  std::cout << "Model resolution:       " << resolution << std::endl;
  std::cout << "Model sampling size:    " << model_ss_ << std::endl;
  std::cout << "Scene sampling size:    " << scene_ss_ << std::endl;
  std::cout << "LRF support radius:     " << rf_rad_ << std::endl;
  std::cout << "SHOT descriptor radius: " << descr_rad_ << std::endl;
  std::cout << "Clustering bin size:    " << cg_size_ << std::endl << std::endl;
  
  //
  //  Downsample Clouds to Extract keypoints
  //

  pcl::UniformSampling<PointType> uniform_sampling;
  uniform_sampling.setInputCloud (model);
  uniform_sampling.setRadiusSearch (model_ss_);
  pcl::PointCloud<int> model_keypoints_out;
  uniform_sampling.compute (model_keypoints_out);
  for (auto k=model_keypoints_out.begin(); k!=model_keypoints_out.end(); k++){
    model_keypoints->push_back(model->at(*k));
  }

  std::cout << "Model total points: " << model->size () << "; Selected Keypoints: " << model_keypoints->size () << std::endl;

  //
  //  Compute Descriptor for keypoints
  //
  pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
  descr_est.setRadiusSearch (descr_rad_);
  descr_est.setInputCloud (model_keypoints);
  descr_est.setInputNormals (model_normals);
  descr_est.setSearchSurface (model);
  descr_est.compute (*model_descriptors);


  //
  //  Set up Kinect frame subscription
  //
  KinectHandlerState kinectHandlerState;
  auto kinect_frame_sub = lcm->subscribeFunction("KINECT_FRAME", &handleKinectFrameMsg, &kinectHandlerState);
  kinect_frame_sub->setQueueCapacity(1); // ensures we don't get delays due to slow processing

  double last_update_time = getUnixTime();
  double timestep = 0.01;
  if (config["timestep"])
    timestep = config["timestep"].as<double>();

  while(1){
    for (int i=0; i < 1; i++)
      lcm->handleTimeout(0);

    double dt = getUnixTime() - last_update_time;
    if (dt > timestep){
      last_update_time = getUnixTime();
      printf("Update!\n");
    } else {
      usleep(1000);
    }
  }

  return 0;
}