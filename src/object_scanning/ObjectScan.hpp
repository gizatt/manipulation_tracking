#ifndef OBJECT_SCAN_H
#define OBJECT_SCAN_H

#include <stdexcept>
#include <iostream>
#include <memory>
#include <mutex>
#include "yaml-cpp/yaml.h"
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <Eigen/Dense>


class ObjectScan {
  typedef pcl::PointXYZ PointType;
  
  public:
    ObjectScan(YAML::Node config);
    ~ObjectScan() {};
    void addPointCloud(pcl::PointCloud<PointType>::Ptr new_pts);
    pcl::PointCloud<pcl::PointXYZ>::Ptr getPointCloud() { return _object_scan; }
  private:
  	pcl::PointCloud<pcl::PointXYZ>::Ptr _object_scan;

};

#endif