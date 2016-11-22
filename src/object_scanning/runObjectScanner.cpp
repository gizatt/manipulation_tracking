#include <stdexcept>
#include <iostream>

#include "common/common.hpp"

#include "ObjectScan.hpp"

#include "drake/systems/plants/RigidBodyTree.h"

#include <lcm/lcm-cpp.hpp>

#include "yaml-cpp/yaml.h"
#include "common/common.hpp"
#include "unistd.h"
#include <mutex>

#include "lcmtypes/kinect/frame_msg_t.hpp"
#include "lcmtypes/bot_core/image_t.hpp"
#include "lcmtypes/bot_core/rigid_transform_t.hpp"
#include <kinect/kinect-utils.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/keyboard_event.h>
#include <pcl/common/transforms.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "zlib.h"

using namespace std;
using namespace Eigen;

typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;

pcl::PointCloud<PointType>::Ptr scene (new pcl::PointCloud<PointType> ());
int downsample_step = 4; // take every 4th pixel
pcl::PointCloud<PointType>::Ptr scene_downsampled (new pcl::PointCloud<PointType> ());

// state and configuration
bool verbose_ = false;
bool add_next_pointcloud_to_object_model = false;
bool do_reset_and_calibration = false;
bool calibrated = false;

// Kinect frame streaming
double lastKinectReceivedTime = -1.0;
std::mutex latest_cloud_mutex;
bool have_unprocessed_kinect_cloud = false;
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> latest_depth_image;
Eigen::Matrix<double, 3, Eigen::Dynamic> latest_color_image;
KinectCalibration* kcal;

// Apriltag streaming
double lastApriltagReceivedTime = -1.0;
std::mutex latest_apriltag_mutex;
bool received_apriltags = false;
struct ApriltagAttachment {
  int list_id;
  float edge_length = 0.1;
  Eigen::Transform<double, 3, Eigen::Isometry> last_transform;
  double last_received;
  lcm::Subscription * detection_sub;
};
std::map<int, ApriltagAttachment> attachedApriltags;
std::map<std::string, int> channelToApriltagIndex;

// Calibration
struct ApriltagCalibration {
  ApriltagAttachment * apriltagAttachment;
  Eigen::Transform<double, 3, Eigen::Isometry> tagToScanOrigin;
};
std::vector<ApriltagCalibration> calibratedApriltags;

double bb_px = 0.1; // right = +x
double bb_nx = -0.1; // left = -x
double bb_ny = -0.1; // back = -y
double bb_py = 0.1; // fwd = +y
double bb_nz = -0.2; // lower = -z
double bb_pz = 0.0; // upper = +z
bool show_object_model = false;
bool do_save = false;

void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                            void* viewer_void)
{
  pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
  if (event.getKeySym () == "d" && event.keyDown ())
  {
    std::cout << "a was pressed -> adding pointcloud to object model" << std::endl;
    add_next_pointcloud_to_object_model = true;
  } else if (event.getKeySym () == "r" && event.keyDown ())
  {
    std::cout << "r was pressed -> capturing apriltag configuration and zeroing camera" << std::endl;
    do_reset_and_calibration = true;
  } else if (event.getKeySym () == "v" && event.keyDown ())
  {
    show_object_model = !show_object_model;
  } else if (event.getKeySym () == "w" && event.keyDown ())
  {
    bb_pz -= 0.002;
  } else if (event.getKeySym () == "s" && event.keyDown ())
  {
    bb_pz += 0.002;
  } else if (event.getKeySym () == "a" && event.keyDown ())
  {
    do_save = true;
  }
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

    if (scene->size() != msg->depth.width * msg->depth.height){
      scene->clear();
      scene_downsampled->clear();

      scene->width = msg->depth.width;
      scene->height = msg->depth.height;
      scene->resize(scene->width * scene->height);
      scene->is_dense = true; // we won't add inf / nan points

      scene_downsampled->width = msg->depth.width / downsample_step;
      scene_downsampled->height = msg->depth.height / downsample_step;
      scene_downsampled->resize(scene_downsampled->width * scene_downsampled->height);
      scene_downsampled->is_dense = true;
    }

    // NB: no depth return is given 0 range - and becomes 0,0,0 here
    if (latest_depth_image.cols() != msg->depth.width || latest_depth_image.rows() != msg->depth.height)
      latest_depth_image.resize(msg->depth.height, msg->depth.width);

    latest_depth_image.setZero();
    for(long int v=0; v<msg->depth.height; v++) { // t2b
      for(long int u=0; u<msg->depth.width; u++ ) {  //l2r 
        // not dealing with color yet

        double constant = 1.0f / kcal->intrinsics_rgb.fx ;
        double disparity_d = depth_data[v*msg->depth.width+u]  / 1000.; // convert to m
        latest_depth_image(v, u) = disparity_d;

        PointType new_point;
        if (!std::isnan(disparity_d) && disparity_d > 0.0 && disparity_d < 4.0){
          new_point.x = (((double) u)- kcal->intrinsics_depth.cx)*disparity_d*constant;  //x right+
          new_point.y = (((double) v)- kcal->intrinsics_depth.cy)*disparity_d*constant; //y down+
          new_point.z = disparity_d; // z forward +
        } else {
          new_point.x = NAN;
          new_point.y = NAN;
          new_point.z = NAN;
        }

        scene->at(u, v) = new_point;
        if (v % 4 == 0 && u % 4 == 0)
          scene_downsampled->at(u/4, v/4) = new_point;
      }
    }
    have_unprocessed_kinect_cloud = true;
  } else {
    printf("Can't unpack different Kinect data format yet.\n");
  }
  latest_cloud_mutex.unlock();
  lastKinectReceivedTime = getUnixTime();
}

class TagDetectionHandlerState{
  public:
    lcm::LCM lcm;
};
void handleTagDetectionMsg(const lcm::ReceiveBuffer* rbuf,
                           const std::string& chan,
                           const bot_core::rigid_transform_t* msg,
                           TagDetectionHandlerState* state){
  auto it = channelToApriltagIndex.find(chan);
  if (it == channelToApriltagIndex.end()){
    printf("Received message from channel we didn't subscribe to... panicking now.\n");
    exit(-1);
  }

  latest_apriltag_mutex.lock();
  ApriltagAttachment * attachment = &attachedApriltags[it->second];
  attachment->last_received = getUnixTime();

  Quaterniond rot(msg->quat[0], msg->quat[1], msg->quat[2], msg->quat[3]);
  attachment->last_transform.setIdentity();
  attachment->last_transform.matrix().block<3, 3>(0,0) = rot.matrix();
  attachment->last_transform.matrix().block<3, 1>(0,3) = Vector3d(msg->trans[0], msg->trans[1], msg->trans[2]);

  received_apriltags = true;
  latest_apriltag_mutex.unlock();
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

void setup_apriltag_subscriber(std::shared_ptr<lcm::LCM> lcm, YAML::Node apriltags_config){
  int numtags = 0;
  for (auto iter=apriltags_config.begin(); iter!=apriltags_config.end(); iter++){
    ApriltagAttachment attachment;

    attachment.list_id = numtags;
    numtags++;

    attachment.last_received = -1.0;
    std::string channel = (*iter)["channel"].as<string>();

    if ((*iter)["edge_length"])
      attachment.edge_length = (*iter)["edge_length"].as<float>();

    TagDetectionHandlerState tagDetectionHandlerState;
    attachment.detection_sub = lcm->subscribeFunction(channel, &handleTagDetectionMsg, &tagDetectionHandlerState);
    attachment.detection_sub->setQueueCapacity(1);

    int id = (*iter)["id"].as<double>();
    attachedApriltags[id] = attachment;

    channelToApriltagIndex[channel] = id;
  }
}

int main(int argc, char** argv) {
  if (argc != 2){
    printf("Use: runObjectScanner <path to yaml config file or \"none\">\n");
    return -1;
  }

  std::shared_ptr<lcm::LCM> lcm(new lcm::LCM());
  if (!lcm->good()) {
    throw std::runtime_error("LCM is not good");
  }

  setup_kinect_calib();

  //
  // Load configuration
  //
  string configFile(argv[1]);


  YAML::Node config;
  if (configFile != "none"){
    printf("Found config file %s\n", configFile.c_str());
    config = YAML::LoadFile(configFile); 
  }

  if (config["apriltags"]){
    printf("Setting up apriltag subscribers\n");
    setup_apriltag_subscriber(lcm, config["apriltags"]);
  }

//
  //  Set up Kinect frame subscription
  //
  KinectHandlerState kinectHandlerState;
  auto kinect_frame_sub = lcm->subscribeFunction("KINECT_FRAME", &handleKinectFrameMsg, &kinectHandlerState);
  kinect_frame_sub->setQueueCapacity(1); // ensures we don't get delays due to slow processing

  double last_update_time = getUnixTime();
  double timestep = 0.5;
  if (config["timestep"])
    timestep = config["timestep"].as<double>();

  //
  // Set up viewer
  //
  pcl::visualization::PCLVisualizer viewer("Object Scanner");
  viewer.registerKeyboardCallback (keyboardEventOccurred, &viewer);

  //
  // Set up object scanner
  // 
  ObjectScan objectScanner(config);

  while(1){
    for (int i=0; i < 1; i++)
      lcm->handleTimeout(0);

    double dt = getUnixTime() - last_update_time;
    if (dt > timestep){
      last_update_time = getUnixTime();

      latest_cloud_mutex.lock();
      latest_apriltag_mutex.lock();

      if (do_reset_and_calibration){
        calibrated = false;
        bb_px = 0.1; // right = +x
        bb_nx = -0.1; // left = -x
        bb_ny = -0.1; // back = -y
        bb_py = 0.1; // fwd = +y
        bb_nz = -0.2; // lower = -z
        bb_pz = 0.0; // upper = +z
        calibratedApriltags.clear();
        if (!received_apriltags){
          printf("Can't calibrate -- no apriltags have ever been detected.\n");
        } else {
          // try calibration procedure
          printf("Calibrating.\n");

          // scan area is centered at the centroid of the tag positions
          // extends in tag avg x, y, and -z (up) dirs by parameter amounts
          // find these values by finding average transforms of the apriltags
          std::vector<Transform<double, 3, Eigen::Isometry>> transforms;

          // go through all attached apriltags, find up-to-date ones, grab them, and also
          // update average transform
          for (auto iter=attachedApriltags.begin(); iter!=attachedApriltags.end(); iter++){
            ApriltagAttachment * attachment = &((*iter).second);
            if (fabs(getUnixTime() - attachment->last_received) < 1.0){
              ApriltagCalibration new_calib;
              new_calib.apriltagAttachment = attachment;
              calibratedApriltags.push_back(new_calib);
              transforms.push_back(attachment->last_transform);
            }
          }
          
          if (transforms.size() == 0){
            printf("Can't calibrate -- no apriltags are up to date.\n");
            calibrated = false;
          } else {
            Eigen::Transform<double, 3, Eigen::Isometry> scanOriginToCamera = getAverageTransform(transforms);
            // Go back through the calibration tags and update their transforms
            for (auto calibrationTag=calibratedApriltags.begin(); calibrationTag!=calibratedApriltags.end(); calibrationTag++){
              // Given last_transform, which encodes tagToCamera
              // and scanOriginToScamera
              // want to recover the intermediate transform, tagToScanOrigin 
              calibrationTag->tagToScanOrigin = calibrationTag->apriltagAttachment->last_transform * (scanOriginToCamera.inverse());
              // and expand bounding box to include origin of this tag, if desired
              Vector3d trans = calibrationTag->tagToScanOrigin.matrix().block<3,1>(0, 3);
              bb_px = fmax(bb_px, trans[0]);
              bb_nx = fmin(bb_nx, trans[0]);
              bb_py = fmax(bb_py, trans[1]);
              bb_ny = fmin(bb_ny, trans[1]);
            }

            calibrated = true;
          }
        }
        do_reset_and_calibration = false;
      }

      // transform scene pointcloud using calibration, if available
      Eigen::Transform<double, 3, Eigen::Isometry> calibration_transform;
      calibration_transform.setIdentity();
      if (calibrated){
        // compute avg implied transform over all apriltags
        std::vector<Transform<double, 3, Eigen::Isometry>> transforms;
        for (auto calibrationTag=calibratedApriltags.begin(); calibrationTag!=calibratedApriltags.end(); calibrationTag++){
          ApriltagAttachment * attachment = calibrationTag->apriltagAttachment;
           if (fabs(getUnixTime() - attachment->last_received) < 1.0){
            // last_transform == tagToCamera
            // we want cameraToOrigin
            Transform<double, 3, Eigen::Isometry> implied_transform = attachment->last_transform.inverse() * calibrationTag->tagToScanOrigin;
            transforms.push_back(implied_transform);
          }
        }
        if (transforms.size() == 0){
          printf("No visible tags, can't get transform\n");
        } else {
          calibration_transform = getAverageTransform(transforms);
        }
      }

      pcl::PointCloud<pcl::PointXYZ>::Ptr scene_calibrated (new pcl::PointCloud<pcl::PointXYZ> ());
      pcl::PointCloud<pcl::PointXYZ>::Ptr scene_calibrated_cropped (new pcl::PointCloud<pcl::PointXYZ> ());
      pcl::transformPointCloud (*scene_downsampled, *scene_calibrated, calibration_transform.matrix()); 

      for (auto it=scene_calibrated->begin(); it!=scene_calibrated->end(); it++){
        if ((*it).x >= bb_nx && (*it).x <= bb_px &&
            (*it).y >= bb_ny && (*it).y <= bb_py &&
            (*it).z >= bb_nz && (*it).z <= bb_pz){
          scene_calibrated_cropped->push_back(*it);
        }
      }

      if (add_next_pointcloud_to_object_model){
        if (!have_unprocessed_kinect_cloud){
          printf("Can't add scan -- no unprocessed kinect pointcloud.\n");
        } else if (!received_apriltags){
          printf("Can't add scan -- no apriltags have ever been detected\n");
        } else if (!calibrated){
          printf("Can't add scan -- not calibrated yet. Get all tags in view then press R.");
        } else {
          printf("Adding an object scan...\n");

          pcl::PointCloud<pcl::PointXYZ>::Ptr scene_calibrated_full (new pcl::PointCloud<pcl::PointXYZ> ());
          pcl::PointCloud<pcl::PointXYZ>::Ptr object_scan (new pcl::PointCloud<pcl::PointXYZ> ());
          pcl::transformPointCloud (*scene, *scene_calibrated_full, calibration_transform.matrix()); 

          for (auto it=scene_calibrated_full->begin(); it!=scene_calibrated_full->end(); it++){
            if ((*it).x >= bb_nx && (*it).x <= bb_px &&
                (*it).y >= bb_ny && (*it).y <= bb_py &&
                (*it).z >= bb_nz && (*it).z <= bb_pz){
              object_scan->push_back(*it);
            }
          }
          objectScanner.addPointCloud(object_scan);
          printf("\tdone.\n");
        }
        add_next_pointcloud_to_object_model = false;
      }

      if (do_save){
        pcl::PointCloud<pcl::PointXYZ>::Ptr objectScan = objectScanner.getPointCloud();
        char filename[128];
        time_t    caltime;
        struct tm * broketime;
        time(&caltime);
        broketime = localtime(&caltime);
        strftime(filename,128,"objectscan_%y%m%d_%H%M%S.pcd",broketime);

        pcl::io::savePCDFileASCII (filename, *objectScan);
        printf("Saved out to %s\n", filename);
        do_save = false;
      }

      // 
      // Do a visualization cycle 
      //
      viewer.removeAllPointClouds();
      viewer.removeAllShapes();
      
      // Add core point cloud
      //viewer.addPointCloud (scene, "scene_cloud");

      if (show_object_model){
        pcl::PointCloud<pcl::PointXYZ>::Ptr objectScan = objectScanner.getPointCloud();
        pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_calibrated_color_handler (objectScan, 255, 100, 100);      
        viewer.addPointCloud (objectScan, scene_calibrated_color_handler, "object_model");
      } else {
        pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_calibrated_color_handler (scene_calibrated_cropped, 100, 255, 100);      
        viewer.addPointCloud (scene_calibrated_cropped, scene_calibrated_color_handler, "scene_cloud_downsampled__calibrated");
      }

      // Draw bounding box around center
      viewer.addLine(PointType(bb_px, bb_ny, bb_nz), PointType(bb_px, bb_py, bb_nz), "l1");
      viewer.addLine(PointType(bb_px, bb_py, bb_nz), PointType(bb_nx, bb_py, bb_nz), "l2");
      viewer.addLine(PointType(bb_nx, bb_py, bb_nz), PointType(bb_nx, bb_ny, bb_nz), "l3");
      viewer.addLine(PointType(bb_nx, bb_ny, bb_nz), PointType(bb_px, bb_ny, bb_nz), "l4");
      viewer.addLine(PointType(bb_px, bb_ny, bb_pz), PointType(bb_px, bb_py, bb_pz), "l5");
      viewer.addLine(PointType(bb_px, bb_py, bb_pz), PointType(bb_nx, bb_py, bb_pz), "l6");
      viewer.addLine(PointType(bb_nx, bb_py, bb_pz), PointType(bb_nx, bb_ny, bb_pz), "l7");
      viewer.addLine(PointType(bb_nx, bb_ny, bb_pz), PointType(bb_px, bb_ny, bb_pz), "l8");
      viewer.addLine(PointType(bb_px, bb_ny, bb_nz), PointType(bb_px, bb_ny, bb_pz), "l9");
      viewer.addLine(PointType(bb_px, bb_py, bb_nz), PointType(bb_px, bb_py, bb_pz), "l10");
      viewer.addLine(PointType(bb_nx, bb_py, bb_nz), PointType(bb_nx, bb_py, bb_pz), "l11");
      viewer.addLine(PointType(bb_nx, bb_ny, bb_nz), PointType(bb_nx, bb_ny, bb_pz), "l12");
      viewer.addLine(PointType(bb_px, bb_py, bb_nz), PointType(bb_nx, bb_ny, bb_nz), "x1");
      viewer.addLine(PointType(bb_px, bb_ny, bb_nz), PointType(bb_nx, bb_py, bb_nz), "x2");
      viewer.addLine(PointType(bb_px, bb_py, bb_pz), PointType(bb_nx, bb_ny, bb_pz), "x3");
      viewer.addLine(PointType(bb_px, bb_ny, bb_pz), PointType(bb_nx, bb_py, bb_pz), "x4");

      // Add apriltags as squares with arrows
      pcl::PointCloud<pcl::PointXYZ>::Ptr box_cloud_raw (new pcl::PointCloud<pcl::PointXYZ> ());
      box_cloud_raw->clear();
      box_cloud_raw->push_back(PointType(-1.0, -1.0, 0.0));
      box_cloud_raw->push_back(PointType(-1.0, 1.0, 0.0));
      box_cloud_raw->push_back(PointType(1.0, 1.0, 0.0));
      box_cloud_raw->push_back(PointType(1.0, -1.0, 0.0));
/*
      for (auto iter=attachedApriltags.begin(); iter!=attachedApriltags.end(); iter++){
        auto attachment = &((*iter).second);
        if (fabs(getUnixTime() - attachment->last_received) < 1.0){
          Vector3d pt = attachment->last_transform* Vector3d::Zero();
          Vector3d arrow_px = attachment->last_transform*Vector3d(attachment->edge_length/2, 0.0, 0.0);
          Vector3d arrow_py = attachment->last_transform*Vector3d(0.0, attachment->edge_length/2, 0.0);

          PointType pclpt(pt[0], pt[1], pt[2]);
          PointType pclarrow_px(arrow_px[0], arrow_px[1], arrow_px[2]);
          PointType pclarrow_py(arrow_py[0], arrow_py[1], arrow_py[2]);

          string thisname = to_string((*iter).first);
          viewer.addText3D(thisname, pclpt, 0.05, 0.2, 0.2, 1.0, "text3d_" + thisname);

          pcl::PointCloud<pcl::PointXYZ>::Ptr box_cloud_transformed (new pcl::PointCloud<pcl::PointXYZ> ());
          Eigen::Affine3d transform = attachment->last_transform;
          transform.matrix().block<3,3>(0,0) = transform.matrix().block<3,3>(0,0) * attachment->edge_length;
          pcl::transformPointCloud (*box_cloud_raw, *box_cloud_transformed, transform); 
          viewer.addPolygon<pcl::PointXYZ> (box_cloud_transformed, 0.2, 0.2, 1.0, "polygon_" + thisname);
          viewer.addArrow(pclarrow_px, pclpt, 1.0, 0.2, 0.2, false, "arrowpx_" + thisname);
          viewer.addArrow(pclarrow_py, pclpt, 0.2, 1.0, 0.2, false, "arrowpy_" + thisname);
        }
      }
*/


      viewer.addText("r: reset calib", 10, 10, "help_text_r");
      viewer.addText("d: add points to object", 10, 30, "help_text_d");
      viewer.addText("v: view object model", 10, 50, "help_text_v");
      viewer.addText("w/s: change bb top", 10, 70, "help_text_bb");
      viewer.addText("a: save object pcd", 10, 90, "help_text_save");

      latest_cloud_mutex.unlock();
      latest_apriltag_mutex.unlock();

      viewer.spinOnce ();

    } else {
      viewer.spinOnce ();
      usleep(1000);
    }
  }

  return 0;
}