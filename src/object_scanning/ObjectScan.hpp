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
    void addPointCloud(pcl::PointCloud<PointType>::Ptr new_pts, Eigen::Affine3d transform);
  private:

};

#endif