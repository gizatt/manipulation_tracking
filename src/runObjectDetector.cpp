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
#include <pcl/features/integral_image_normal.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/board.h>
#include <pcl/keypoints/uniform_sampling.h>
#include <pcl/recognition/cg/hough_3d.h>
#include <pcl/recognition/cg/geometric_consistency.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/keyboard_event.h>
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
pcl::PointCloud<PointType>::Ptr scene_pruned (new pcl::PointCloud<PointType> ());
pcl::PointCloud<PointType>::Ptr scene_keypoints (new pcl::PointCloud<PointType> ());
pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<NormalType>::Ptr scene_normals_pruned (new pcl::PointCloud<NormalType> ());
pcl::PointCloud<DescriptorType>::Ptr model_descriptors (new pcl::PointCloud<DescriptorType> ());
pcl::PointCloud<DescriptorType>::Ptr scene_descriptors (new pcl::PointCloud<DescriptorType> ());

pcl::CorrespondencesPtr model_scene_corrs (new pcl::Correspondences ());

std::vector<Eigen::Matrix4f, Eigen::aligned_allocator<Eigen::Matrix4f> > rototranslations;
std::vector<pcl::Correspondences> clustered_corrs;

//Algorithm params
bool show_keypoints_ (true);
bool show_correspondences_ (true);
bool use_cloud_resolution_ (false);
bool use_hough_ (true);
float model_ss_ (0.025f);
float scene_ss_ (0.025f);
float rf_rad_ (0.015f);
float descr_rad_ (0.02f);
float cg_size_ (0.01f);
float cg_thresh_ (5.0f);
float shot_dist_thresh_ (0.25);
bool verbose_ (true);

bool do_object_detection = false;

// Kinect frame streaming
double lastKinectReceivedTime = -1.0;
std::mutex latest_cloud_mutex;
bool have_unprocessed_kinect_cloud = false;
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> latest_depth_image;
Eigen::Matrix<double, 3, Eigen::Dynamic> latest_color_image;
KinectCalibration* kcal;

void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                            void* viewer_void)
{
  pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
  if (event.getKeySym () == "d" && event.keyDown ())
  {
    std::cout << "d was pressed => kicking off detection" << std::endl;
    do_object_detection = true;
  }
}

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

    scene->width = msg->depth.width;
    scene->height = msg->depth.height;
    scene->resize(scene->width * scene->height);
    scene->is_dense = true; // we won't add inf / nan points

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
        if (!std::isnan(disparity_d) && disparity_d > 0.0 && disparity_d < 1.0){
          new_point.x = (((double) u)- kcal->intrinsics_depth.cx)*disparity_d*constant;  //x right+
          new_point.y = (((double) v)- kcal->intrinsics_depth.cy)*disparity_d*constant; //y down+
          new_point.z = disparity_d; // z forward +
        } else {
          new_point.x = NAN;
          new_point.y = NAN;
          new_point.z = NAN;
        }

        scene->at(u, v) = new_point;
      }
    }
    have_unprocessed_kinect_cloud = true;
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

  //
  // Load configuration
  //
  char * configFile = argv[1];
  YAML::Node config = YAML::LoadFile(string(configFile)); 

  if (config["shot_dist_thresh"])
    shot_dist_thresh_ = config["shot_dist_thresh"].as<float>();
  if (config["model_ss"])
    model_ss_ = config["model_ss"].as<float>();
  if (config["scene_ss"])
    scene_ss_ = config["scene_ss"].as<float>();
  if (config["rf_rad"])
    rf_rad_ = config["rf_rad"].as<float>();
  if (config["descr_rad"])
    descr_rad_ = config["descr_rad"].as<float>();
  if (config["cg_size"])
    cg_size_ = config["cg_size"].as<float>();
  if (config["cg_thresh"])
    cg_thresh_ = config["cg_thresh"].as<float>();
  if (config["verbose"])
    verbose_ = config["verbose"].as<bool>();

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
  //  Set up Kinect frame subscription
  //
  KinectHandlerState kinectHandlerState;
  auto kinect_frame_sub = lcm->subscribeFunction("KINECT_FRAME", &handleKinectFrameMsg, &kinectHandlerState);
  kinect_frame_sub->setQueueCapacity(1); // ensures we don't get delays due to slow processing

  double last_update_time = getUnixTime();
  double timestep = 0.01;
  if (config["timestep"])
    timestep = config["timestep"].as<double>();

  //
  // Set up viewer
  //
  pcl::visualization::PCLVisualizer viewer("Object Detection");
  bool visualizer_has_scene_cloud = false;
  viewer.registerKeyboardCallback (keyboardEventOccurred, &viewer);

  while(1){
    for (int i=0; i < 1; i++)
      lcm->handleTimeout(0);

    double dt = getUnixTime() - last_update_time;
    if (dt > timestep){
      last_update_time = getUnixTime();
      printf("Update!\n");

      latest_cloud_mutex.lock();
      if (have_unprocessed_kinect_cloud){
        scene_pruned->clear();
        scene_normals->clear();
        scene_normals_pruned->clear();

        // do feature detection in pointcloud

        //
        //  Compute Normals
        //
        pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> norm_est;
        norm_est.setNormalEstimationMethod (norm_est.AVERAGE_3D_GRADIENT);
        norm_est.setMaxDepthChangeFactor(0.02f);
        norm_est.setNormalSmoothingSize(10.0f);
        norm_est.setInputCloud(scene);
        norm_est.compute (*scene_normals);

        // Prune out all points and normals in scene that are NAN

        if (scene->size() != scene_normals->size()){
          printf("Scene size %ld but scene normals size %ld. What gives?\n", scene->size(), scene_normals->size());
          return -1;
        }
        scene_pruned->resize(scene->size());
        scene_normals_pruned->resize(scene_normals->size());

        int k_good = 0;
        for (int k=0; k<scene->size(); k++){
          PointType this_pt = scene->at(k);
          NormalType this_norm = scene_normals->at(k);

          if (!(isnan(this_pt.x) || isnan(this_norm.normal_x))){
            scene_pruned->at(k_good) = this_pt;
            scene_normals_pruned->at(k_good) = this_norm;
            k_good++;
          }
        }
        scene_pruned->resize(k_good);
        scene_normals_pruned->resize(k_good);


        if (do_object_detection){
          scene_keypoints->clear();
          scene_descriptors->clear();
          model_scene_corrs->clear();
          rototranslations.clear();
          clustered_corrs.clear();

          //
          //  Downsample Clouds to Extract keypoints
          //
          uniform_sampling.setInputCloud (scene_pruned);
          uniform_sampling.setRadiusSearch (scene_ss_);
          pcl::PointCloud<int> scene_keypoints_out;
          uniform_sampling.compute (scene_keypoints_out);
          for (auto k=scene_keypoints_out.begin(); k!=scene_keypoints_out.end(); k++){
            scene_keypoints->push_back(scene_pruned->at(*k));
          }
          std::cout << "Scene total points: " << scene->size () << "; Pruned total points: " << scene_pruned->size() << "; Selected Keypoints: " << scene_keypoints->size () << std::endl;

          //
          //  Compute Descriptor for keypoints
          //
          pcl::SHOTEstimationOMP<PointType, NormalType, DescriptorType> descr_est;
          descr_est.setRadiusSearch (descr_rad_);
          descr_est.setInputCloud (model_keypoints);
          descr_est.setInputNormals (model_normals);
          descr_est.setSearchSurface (model);
          descr_est.compute (*model_descriptors);

          descr_est.setInputCloud (scene_keypoints);
          descr_est.setInputNormals (scene_normals_pruned);
          descr_est.setSearchSurface (scene_pruned);
          descr_est.compute (*scene_descriptors);

          //
          //  Find Model-Scene Correspondences with KdTree
          //
          pcl::KdTreeFLANN<DescriptorType> match_search;
          match_search.setInputCloud (model_descriptors);

          //  For each scene keypoint descriptor, find nearest neighbor into the model keypoints descriptor cloud and add it to the correspondences vector.
          for (size_t i = 0; i < scene_descriptors->size (); ++i)
          {
            std::vector<int> neigh_indices (1);
            std::vector<float> neigh_sqr_dists (1);
            if (!pcl_isfinite (scene_descriptors->at (i).descriptor[0])) //skipping NaNs
            {
              continue;
            }
            int found_neighs = match_search.nearestKSearch (scene_descriptors->at (i), 1, neigh_indices, neigh_sqr_dists);
            printf("dist %f, ", neigh_sqr_dists[0]);
            if(found_neighs == 1 && neigh_sqr_dists[0] < shot_dist_thresh_) //  add match only if the squared descriptor distance is less than 0.25 (SHOT descriptor distances are between 0 and 1 by design)
            {
              pcl::Correspondence corr (neigh_indices[0], static_cast<int> (i), neigh_sqr_dists[0]);
              model_scene_corrs->push_back (corr);
            }
          }
          std::cout << "Correspondences found: " << model_scene_corrs->size () << std::endl;

          //
          //  Actual Clustering
          //

          //  Using Hough3D
          if (use_hough_)
          {
            //
            //  Compute (Keypoints) Reference Frames only for Hough
            //
            pcl::PointCloud<RFType>::Ptr model_rf (new pcl::PointCloud<RFType> ());
            pcl::PointCloud<RFType>::Ptr scene_rf (new pcl::PointCloud<RFType> ());

            pcl::BOARDLocalReferenceFrameEstimation<PointType, NormalType, RFType> rf_est;
            rf_est.setFindHoles (true);
            rf_est.setRadiusSearch (rf_rad_);

            rf_est.setInputCloud (model_keypoints);
            rf_est.setInputNormals (model_normals);
            rf_est.setSearchSurface (model);
            rf_est.compute (*model_rf);

            rf_est.setInputCloud (scene_keypoints);
            rf_est.setInputNormals (scene_normals_pruned);
            rf_est.setSearchSurface (scene_pruned);
            rf_est.compute (*scene_rf);

            //  Clustering
            pcl::Hough3DGrouping<PointType, PointType, RFType, RFType> clusterer;
            clusterer.setHoughBinSize (cg_size_);
            clusterer.setHoughThreshold (cg_thresh_);
            clusterer.setUseInterpolation (true);
            clusterer.setUseDistanceWeight (false);

            clusterer.setInputCloud (model_keypoints);
            clusterer.setInputRf (model_rf);
            clusterer.setSceneCloud (scene_keypoints);
            clusterer.setSceneRf (scene_rf);
            clusterer.setModelSceneCorrespondences (model_scene_corrs);

            //clusterer.cluster (clustered_corrs);
            clusterer.recognize (rototranslations, clustered_corrs);
          }
          else // Using GeometricConsistency
          {
            pcl::GeometricConsistencyGrouping<PointType, PointType> gc_clusterer;
            gc_clusterer.setGCSize (cg_size_);
            gc_clusterer.setGCThreshold (cg_thresh_);

            gc_clusterer.setInputCloud (model_keypoints);
            gc_clusterer.setSceneCloud (scene_keypoints);
            gc_clusterer.setModelSceneCorrespondences (model_scene_corrs);

            //gc_clusterer.cluster (clustered_corrs);
            gc_clusterer.recognize (rototranslations, clustered_corrs);
          }

          //
          //  Output results
          //
          std::cout << "Model instances found: " << rototranslations.size () << std::endl;
          for (size_t i = 0; i < rototranslations.size (); ++i)
          {
            std::cout << "\n    Instance " << i + 1 << ":" << std::endl;
            std::cout << "        Correspondences belonging to this instance: " << clustered_corrs[i].size () << std::endl;

            // Print the rotation matrix and translation vector
            Eigen::Matrix3f rotation = rototranslations[i].block<3,3>(0, 0);
            Eigen::Vector3f translation = rototranslations[i].block<3,1>(0, 3);

            printf ("\n");
            printf ("            | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
            printf ("        R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
            printf ("            | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
            printf ("\n");
            printf ("        t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));
          }
          do_object_detection = false;
        }

        have_unprocessed_kinect_cloud = false;
      }


      // 
      // Do a visualization cycle 
      //
      viewer.removeAllPointClouds();
      viewer.removeAllShapes();
      viewer.addPointCloud (scene_pruned, "scene_cloud");
      
      pcl::PointCloud<PointType>::Ptr off_scene_model (new pcl::PointCloud<PointType> ());
      pcl::PointCloud<PointType>::Ptr off_scene_model_keypoints (new pcl::PointCloud<PointType> ());

      if (show_correspondences_ || show_keypoints_)
      {
        //  We are translating the model so that it doesn't end in the middle of the scene representation
        pcl::transformPointCloud (*model, *off_scene_model, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));
        pcl::transformPointCloud (*model_keypoints, *off_scene_model_keypoints, Eigen::Vector3f (-1,0,0), Eigen::Quaternionf (1, 0, 0, 0));

        pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_color_handler (off_scene_model, 255, 255, 128);
        viewer.addPointCloud (off_scene_model, off_scene_model_color_handler, "off_scene_model"); 
      }

      if (show_keypoints_)
      {
        pcl::visualization::PointCloudColorHandlerCustom<PointType> off_scene_model_keypoints_color_handler (off_scene_model_keypoints, 0, 0, 255);
        pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_keypoints_color_handler (scene_keypoints, 0, 0, 255);
          
        viewer.addPointCloud (scene_keypoints, scene_keypoints_color_handler, "scene_keypoints");
        viewer.addPointCloud (off_scene_model_keypoints, off_scene_model_keypoints_color_handler, "off_scene_model_keypoints");

        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "off_scene_model_keypoints");
        viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "scene_keypoints");
      }

      if (show_correspondences_)
      {
        for (size_t j = 0; j < model_scene_corrs->size(); ++j)
        {
          std::stringstream ss_line;
          ss_line << "raw_correspondence_line" << j;
          PointType& model_point = off_scene_model_keypoints->at (model_scene_corrs->at(j).index_query);
          PointType& scene_point = scene_keypoints->at (model_scene_corrs->at(j).index_match);

          //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
          viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
        }
      }
/*
      for (size_t i = 0; i < rototranslations.size (); ++i)
      {
        pcl::PointCloud<PointType>::Ptr rotated_model (new pcl::PointCloud<PointType> ());
        pcl::transformPointCloud (*model, *rotated_model, rototranslations[i]);

        std::stringstream ss_cloud;
        ss_cloud << "instance" << i;

        pcl::visualization::PointCloudColorHandlerCustom<PointType> rotated_model_color_handler (rotated_model, 255, 0, 0);
        viewer.addPointCloud (rotated_model, rotated_model_color_handler, ss_cloud.str ());

        if (show_correspondences_)
        {
          for (size_t j = 0; j < clustered_corrs[i].size (); ++j)
          {
            std::stringstream ss_line;
            ss_line << "correspondence_line" << i << "_" << j;
            PointType& model_point = off_scene_model_keypoints->at (clustered_corrs[i][j].index_query);
            PointType& scene_point = scene_keypoints->at (clustered_corrs[i][j].index_match);

            //  We are drawing a line for each pair of clustered correspondences found between the model and the scene
            viewer.addLine<PointType, PointType> (model_point, scene_point, 0, 255, 0, ss_line.str ());
          }
        }
      }
*/
      visualizer_has_scene_cloud = true;
      latest_cloud_mutex.unlock();

      viewer.spinOnce ();

    } else {
      viewer.spinOnce ();
      usleep(1000);
    }
  }

  return 0;
}