#include <stdexcept>
#include <iostream>

#include "drake/multibody/rigid_body_tree.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/common/eigen_matrix_compare.h"
#include "drake/common/eigen_types.h"

#include <lcm/lcm-cpp.hpp>

#include "yaml-cpp/yaml.h"
#include "common/common.hpp"
#include "unistd.h"

#include <random>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/geometry.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

using namespace std;
using namespace Eigen;
using namespace drake::solvers;

typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;

static inline double randrange(double min, double max){
  return (((double)rand()) / RAND_MAX)*(max - min) + min;
}

double pointDistance(const PointType& pt1, const PointType& pt2){
  return sqrtf(powf(pt1.x - pt2.x, 2) + powf(pt1.y - pt2.y, 2) + powf(pt1.z - pt2.z, 2));
}

#include <time.h>
// call this function to start a nanosecond-resolution timer
struct timespec timer_start(){
    struct timespec start_time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
    return start_time;
}
// call this function to end a timer, returning nanoseconds elapsed as a long
long timer_end(struct timespec start_time){
    struct timespec end_time;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
    long diffInNanos = end_time.tv_nsec - start_time.tv_nsec;
    return diffInNanos;
}

// TODO: replace with reshape?
DecisionVariableMatrixX flatten_3xN( const DecisionVariableMatrixX& x ){
  assert(x.rows() == 3);
  DecisionVariableMatrixX ret(3*x.cols(), 1);
  for (int i=0; i<x.cols(); i++){
    ret.block<3, 1>(i*3, 0) = x.block<3,1>(0, i);
  }
  return ret;
}

int main(int argc, char** argv) {
  srand(getUnixTime());

  if (argc != 4){
    printf("Use: minlp_detector <numrays> <neighborhood size> <urdf>\n");
    exit(-1);
  }

  std::shared_ptr<lcm::LCM> lcm(new lcm::LCM());
  if (!lcm->good()) {
    throw std::runtime_error("LCM is not good");
  }

  int kNumRays = atoi(argv[1]);
  float kSceneNeighborhoodSize = atof(argv[2]);

  // Set up robot
  string urdfString = string(argv[3]);
  RigidBodyTree<double> robot(urdfString);
  KinematicsCache<double> robot_kinematics_cache(robot.bodies);
  VectorXd q0_robot(robot.get_num_positions());
  q0_robot.setZero();
  robot_kinematics_cache.initialize(q0_robot);
  robot.doKinematics(robot_kinematics_cache);
  printf("Set up robot with %d positions\n", robot.get_num_positions());

  // Render scene point cloud
  pcl::PointCloud<PointType>::Ptr model_pts (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr scene_pts (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<NormalType>::Ptr model_normals (new pcl::PointCloud<NormalType> ());
  pcl::PointCloud<NormalType>::Ptr scene_normals (new pcl::PointCloud<NormalType> ());

  Matrix3Xd origins(3, kNumRays);
  Matrix3Xd endpoints(3, kNumRays);
  VectorXd distances(kNumRays);
  Matrix3Xd normals(3, kNumRays);
  for (int k=0; k<kNumRays; k++){
    origins(0, k) = randrange(-100, 100);
    origins(1, k) = randrange(-100, 100);
    origins(2, k) = randrange(-100, 100);

    endpoints(0, k) = -origins(0, k);
    endpoints(1, k) = -origins(1, k);
    endpoints(2, k) = -origins(2, k);
  }
  vector<int> collision_body(kNumRays);
  robot.collisionRaycast(robot_kinematics_cache, origins, endpoints, distances, normals, collision_body, false);

  // Generate model point cloud -- tons of points on surface of the model
  for (int k=0; k<kNumRays; k++){
    if (distances(k) >= 0 && collision_body[k] == 1){
      Vector3d dir = endpoints.block<3, 1>(0, k) - origins.block<3,1>(0, k);
      dir /= dir.norm();
      Vector3d pt = origins.block<3,1>(0, k) + dir * distances(k);
      model_pts->push_back(PointType( pt(0), pt(1), pt(2)));
      model_normals->push_back(NormalType( normals(0, k), normals(1, k), normals(2, k)));
    }
  }
  if (model_pts->size() == 0){
    printf("No points generated for model! Aborting.\n");
    return -1;
  }

  // Generate "scene" point cloud by picking a random point, and taking
  // all points within a neighborhood of it.
  scene_pts->push_back(model_pts->at(0));
  scene_normals->push_back(model_normals->at(0));
  for (int k=1; k<model_pts->size(); k++){
    if (pointDistance(scene_pts->at(0), model_pts->at(k)) < kSceneNeighborhoodSize){
      scene_pts->push_back(model_pts->at(k));
      scene_normals->push_back(model_normals->at(k));
    }
  }

  // Randomly transform scene point cloud
  Eigen::Affine3f scene_model_tf = Eigen::Affine3f::Identity();
  // Define a translation of 2.5 meters on the x axis.
  //scene_model_tf.translation() << randrange(-0.5, 0.5), randrange(-0.5, 0.5), randrange(-0.5, 0.5);
  // The same rotation matrix as before; theta radians arround Z axis
  //scene_model_tf.rotate (Eigen::AngleAxisf (randrange(-1.57, 1.57), Eigen::Vector3f::UnitZ()));

  pcl::PointCloud<PointType>::Ptr scene_pts_tf (new pcl::PointCloud<PointType> ());
  pcl::transformPointCloud (*scene_pts, *scene_pts_tf, scene_model_tf);

  printf("Selected %ld model pts and %ld scene pts\n", model_pts->size(), scene_pts->size());
  printf("Ground truth TF: ");
  cout << scene_model_tf.translation().transpose() << endl;
  cout << scene_model_tf.matrix().block<3,3>(0,0) << endl;
  printf("*******\n");

  pcl::visualization::PCLVisualizer viewer ("Point Collection");

  pcl::visualization::PointCloudColorHandlerCustom<PointType> model_color_handler (model_pts, 128, 255, 255);
  viewer.addPointCloud<PointType>(model_pts, model_color_handler, "model pts"); 
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "model pts");

  pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_color_handler (scene_pts_tf, 255, 255, 128);
  viewer.addPointCloud<PointType>(scene_pts_tf, scene_color_handler, "scene pts tf"); 
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "scene pts tf");




  /*
  // Solves earth movers problem for scene to model cloud
  // Crazy naive mixed integer question we're going to ask is
  // assignment of points in model to points in scene
  MathematicalProgram prog;
  auto C = prog.AddBinaryVariables(scene_pts_tf->size(), model_pts->size(), "c");

  // Every scene (tactile) point should correspond to exactly 1 model point
  Eigen::MatrixXd C1 = Eigen::MatrixXd::Ones(1, model_pts->size());
  for (size_t k=0; k<scene_pts_tf->size(); k++){
    prog.AddLinearConstraint(C1, 1, 1, {C.block(k, 0, 1, model_pts->size()).transpose()});
  }

  // Every model point should correspond to at most 1 scene point
  Eigen::MatrixXd C2 = Eigen::MatrixXd::Ones(1, scene_pts_tf->size());
  for (size_t k=0; k<model_pts->size(); k++){
    prog.AddLinearConstraint(C2, 0, 1, {C.block(0, k, scene_pts_tf->size(), 1)});
  }

  // Every correspondence is weighted by the DISTANCE of that correspondence -- shorter is better
  for (int k_s=0; k_s<scene_pts_tf->size(); k_s++){
    for (int k_m=0; k_m<model_pts->size(); k_m++){
      prog.AddLinearCost( drake::Vector1d(pointDistance(scene_pts_tf->at(k_s), model_pts->at(k_m))), {C.block<1,1>(k_s, k_m)} );
    }
  }
  double now = getUnixTime();
  auto out = prog.Solve();
  string problem_string = "earthmovers";
  double elapsed = getUnixTime() - now;
  */

  //prog.PrintSolution();




  // Solve rigid body transform problem with correspondences

  // We want to solve
  //  \min \sum_{i \in N_s} ||R s_i + T - C_i M||^2
  // R being a 3x3 rotation matrix, T being a 3x1 translation, 
  // C_i being a collection of binary correspondence variables
  // relating scene point i to some model point, and M being
  // the set of model points
  // M = {3*N_m, 1} matrix appending all model points
  // C_i = {3 x 3*N_m} matrix with structure
  //  [ c_{i, 1} * I(3) ... ... ... c_{i, N_m} * I(3) ]
  // 
  // To get things in the right order, we instead minimize
  // \min \sum_{i \in N_s} ||s_i^T R^T + T^T - M^T C_i^T||^2
  // Because || B * x ||^2 = x^T B^T B x
  // and x = [R^T
  //          T^T
  //          C_i^T]
  // B = [s_i^T   1  -M^T]
  // giving us our quadratic cost
  // This is gonna be huge!

  MathematicalProgram prog;

  auto C = prog.AddBinaryVariables(scene_pts_tf->size(), model_pts->size(), "c");
  auto R = prog.AddContinuousVariables(3, 3, "r");
  auto T = prog.AddContinuousVariables(3, 1, "t");


  // constrain T to identity for now
  prog.AddLinearConstraint(Eigen::MatrixXd::Identity(3, 3),
    Eigen::VectorXd::Zero(3), Eigen::VectorXd::Zero(3), 
    {T});

  // constrain R to identity for now
  // using block instead of direct access to convince compiler that we can actually
  // build a variable list from individual elements
  // I'm sure there's a cleaner way but this is really for debugging...
  prog.AddLinearConstraint(Eigen::MatrixXd::Identity(3, 3),
    Eigen::VectorXd::Ones(3), Eigen::VectorXd::Ones(3), 
    {R.block<1,1>(0,0), R.block<1,1>(1,1), R.block<1,1>(2,2)});
  prog.AddLinearConstraint(Eigen::MatrixXd::Identity(6,6),
    Eigen::VectorXd::Zero(6), Eigen::VectorXd::Zero(6), 
    {R.block<1,1>(0,1), R.block<1,1>(0,2), R.block<1,1>(1,0),
     R.block<1,1>(1,2), R.block<1,1>(2,0), R.block<1,1>(2,1)});

  // Every scene (tactile) point should correspond to at most 1 model point
  Eigen::MatrixXd C1 = Eigen::MatrixXd::Ones(1, model_pts->size());
  for (size_t k=0; k<scene_pts_tf->size(); k++){
    prog.AddLinearConstraint(C1, 1, 1, {C.block(k, 0, 1, model_pts->size()).transpose()});
  }

  Eigen::VectorXd M = VectorXd(model_pts->size() * 3);
  for (int j=0; j<model_pts->size(); j++){
    M(3*j + 0) = model_pts->at(j).x;
    M(3*j + 1) = model_pts->at(j).y;
    M(3*j + 2) = model_pts->at(j).z;
  }

  // for debugging, I'm adding a dummy var constrained to zero to 
  // fill out the diagonals of C_i
  auto C_dummy = prog.AddContinuousVariables(1, "c_dummy_zero");
  prog.AddLinearConstraint(Eigen::MatrixXd::Ones(1, 1), Eigen::MatrixXd::Zero(1, 1), Eigen::MatrixXd::Zero(1, 1), {C_dummy});

  printf("THIS FIRST PART IS SKETCHY...\n");
  for (int i=0; i<scene_pts_tf->size(); i++){
    auto C_i = DecisionVariableMatrixX(3, 3*model_pts->size());
    for (int j=0; j<model_pts->size(); j++){
      //C_i.block<3,3>(0, 3*j) ::Zero();
      C_i(0, 3*j) = C(i, j); 
      C_i(1, 3*j+1) = C(i, j);
      C_i(2, 3*j+2) = C(i, j);

      C_i(1, 3*j) = C_dummy(0,0);
      C_i(2, 3*j) = C_dummy(0,0);
      C_i(0, 3*j+1) = C_dummy(0,0);
      C_i(2, 3*j+1) = C_dummy(0,0);
      C_i(0, 3*j+2) = C_dummy(0,0);
      C_i(1, 3*j+2) = C_dummy(0,0);
    }

    // Gotta flatten B to add quadratic cost using column vectors
    auto B = Eigen::RowVectorXd(3*(3+1+3*model_pts->size()));
    B(0) = scene_pts_tf->at(i).x;
    B(1) = scene_pts_tf->at(i).y;
    B(2) = scene_pts_tf->at(i).z;
    B(3) = scene_pts_tf->at(i).x;
    B(4) = scene_pts_tf->at(i).y;
    B(5) = scene_pts_tf->at(i).z;
    B(6) = scene_pts_tf->at(i).x;
    B(7) = scene_pts_tf->at(i).y;
    B(8) = scene_pts_tf->at(i).z;
    B(9) = 1.0;
    B(10) = 1.0;
    B(11) = 1.0;
    B.block(0, 12, 1, model_pts->size()*3) = -1.0 * M.transpose();
    B.block(0, 12+model_pts->size()*3, 1, model_pts->size()*3) = -1.0 * M.transpose();
    B.block(0, 12+2*model_pts->size()*3, 1, model_pts->size()*3) = -1.0 * M.transpose();

    printf("Setting up quadratic cost for scene pt %d, adding %lu new constraints...\n", i, 3*(3+1+3*model_pts->size()));
    prog.AddQuadraticCost(B.transpose() * B, Eigen::VectorXd::Zero(3*(3+1+3*model_pts->size())), {flatten_3xN(R), T, flatten_3xN(C_i)});
  }

  double now = getUnixTime();
  auto out = prog.Solve();
  string problem_string = "rigidtf";
  double elapsed = getUnixTime() - now;

  printf("Transform:\n");
  printf("\tTranslation: %f, %f, %f\n", T(0, 0).value(), T(1, 0).value(), T(2, 0).value());
  printf("\tRotation:\n");
  printf("\t\t%f, %f, %f\n", R(0, 0).value(), R(0, 1).value(), R(0, 2).value());
  printf("\t\t%f, %f, %f\n", R(1, 0).value(), R(1, 1).value(), R(1, 2).value());
  printf("\t\t%f, %f, %f\n", R(2, 0).value(), R(2, 1).value(), R(2, 2).value());


  for (int k_s=0; k_s<scene_pts_tf->size(); k_s++){
    for (int k_m=0; k_m<model_pts->size(); k_m++){
      if (C(k_s, k_m).value() == 1){
        printf("Corresp s%d->m%d at dist %f\n", k_s, k_m, pointDistance(scene_pts_tf->at(k_s), model_pts->at(k_m)));
        std::stringstream ss_line;
        ss_line << "raw_correspondence_line" << k_m << "-" << k_s;
        viewer.addLine<PointType, PointType> (model_pts->at(k_m), scene_pts_tf->at(k_s), 0, 255, 0, ss_line.str ());
      }
    }
  }

  printf("Problem %s solved for %lu scene, %lu model solved in: %f\n", problem_string.c_str(), scene_pts_tf->size(), model_pts->size(), elapsed);

  while (!viewer.wasStopped ())
    viewer.spinOnce ();

  return 0;
}
