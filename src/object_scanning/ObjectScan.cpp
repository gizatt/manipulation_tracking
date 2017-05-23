#include <assert.h>
#include "ObjectScan.hpp"
#include <cmath>
#include "common/common.hpp"
#include <pcl/registration/icp.h>

using namespace std;
using namespace Eigen;

ObjectScan::ObjectScan(YAML::Node config)
{
  if (config["parameter"]){
    printf("now what\n");
  }
  _object_scan = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>());
  _object_scan->clear();
}

void ObjectScan::addPointCloud(pcl::PointCloud<PointType>::Ptr new_pts){

  if (_object_scan->size() > 0){
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaximumIterations (100);
    icp.setInputSource (new_pts);
    icp.setInputTarget (_object_scan);
    icp.align (*new_pts);
    icp.setMaximumIterations (1);  // We set this variable to 1 for the next time we will call .align () function

    if (!icp.hasConverged ()){
    printf("ICP didn't work.\n");
      return;
    } 
  }
  for (auto it=new_pts->begin(); it!=new_pts->end(); it++){
    _object_scan->push_back(*it);
  }
}