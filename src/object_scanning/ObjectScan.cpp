#include <assert.h>
#include "ObjectScan.hpp"
#include <cmath>
#include "common/common.hpp"

using namespace std;
using namespace Eigen;

ObjectScan::ObjectScan(YAML::Node config)
{
  if (config["parameter"]){
    printf("now what\n");
  }
}

void ObjectScan::addPointCloud(pcl::PointCloud<PointType>::Ptr new_pts, Eigen::Affine3d transform){
  printf("Trying\n");
  cout << transform.matrix() << endl;
}