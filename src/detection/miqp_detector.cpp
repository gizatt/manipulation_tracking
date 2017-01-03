/*
 *
 * This detector tackles the 3D-3D point correspondence and transformation estimation
 * problem with a Mixed Integer Quadratic Program.
 *
 * Given a set of scene points s_i and a larger set of model points m_j,
 * i \in [0, N_s] and j \in [0, N_m], we want to find
 * 3x3 rotation matrix R, 3x1 translation matrix T, and N_
 * \sum
 *
 */


#include <stdexcept>
#include <iostream>

#include "drake/multibody/rigid_body_tree.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mosek_solver.h"
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

#include "optimization_helpers.h"
#include "rotation_helpers.h"

using namespace std;
using namespace Eigen;
using namespace drake::solvers;

typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;

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
  vector<int> correspondences_gt;
  correspondences_gt.push_back(0);
  for (int k=1; k<model_pts->size(); k++){
    if (pointDistance(scene_pts->at(0), model_pts->at(k)) < kSceneNeighborhoodSize){
      scene_pts->push_back(model_pts->at(k));
      scene_normals->push_back(model_normals->at(k));
      correspondences_gt.push_back(k);
    }
  }

  // Center scene point cloud
  pcl::PointCloud<PointType>::Ptr scene_pts_tf (new pcl::PointCloud<PointType> ());
  Vector3d avg_scene_pt = Vector3d::Zero();
  for (int k=0; k<scene_pts->size(); k++){
    avg_scene_pt += Vector3d(scene_pts->at(k).x, scene_pts->at(k).y, scene_pts->at(k).z);
  }
  avg_scene_pt /= (double)(scene_pts->size());

  Eigen::Affine3f scene_centering_tf = Eigen::Affine3f::Identity();
  scene_centering_tf.translation() = avg_scene_pt.cast<float>();
  pcl::transformPointCloud (*scene_pts, *scene_pts_tf, scene_centering_tf);

  // Randomly transform model point cloud
  Eigen::Affine3f scene_model_tf = Eigen::Affine3f::Identity();
  scene_model_tf.translation() << randrange(-0.5, 0.5), randrange(-0.5, 0.5), randrange(-0.5, 0.5);
  // theta radians arround Z axis
  scene_model_tf.rotate (Eigen::AngleAxisf (randrange(-1.57, 1.57), Eigen::Vector3f::UnitZ()));

  pcl::PointCloud<PointType>::Ptr model_pts_tf (new pcl::PointCloud<PointType> ());
  pcl::transformPointCloud (*model_pts, *model_pts_tf, scene_model_tf);


  printf("Selected %ld model pts and %ld scene pts\n", model_pts->size(), scene_pts_tf->size());
  printf("Ground truth TF: ");
  cout << -scene_model_tf.translation().transpose() << endl;
  cout << scene_model_tf.matrix().block<3,3>(0,0).inverse() << endl;
  printf("*******\n");

  pcl::visualization::PCLVisualizer viewer ("Point Collection");

  pcl::visualization::PointCloudColorHandlerCustom<PointType> model_color_handler (model_pts_tf, 128, 255, 255);
  viewer.addPointCloud<PointType>(model_pts_tf, model_color_handler, "model pts_tf"); 
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "model pts_tf");

  pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_color_handler (scene_pts_tf, 255, 255, 128);
  viewer.addPointCloud<PointType>(scene_pts_tf, scene_color_handler, "scene pts"); 
  viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "scene pts");


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


  MathematicalProgram prog;

  auto C = prog.AddBinaryVariables(scene_pts_tf->size(), model_pts_tf->size(), "c");
  /*prog.AddBoundingBoxConstraint(VectorXd::Zero(scene_pts_tf->size()*model_pts_tf->size()),
                                VectorXd::Ones(scene_pts_tf->size()*model_pts_tf->size()),
                                {flatten_MxN(C)});*/

  auto T = prog.AddContinuousVariables(3, 1, "t");
  // And add bounding box constraints on appropriate variables
  prog.AddBoundingBoxConstraint(-100*VectorXd::Ones(3), 100*VectorXd::Ones(3), {flatten_MxN(T)});


  auto R = prog.AddContinuousVariables(3, 3, "r");
  prog.AddBoundingBoxConstraint(-VectorXd::Ones(9), VectorXd::Ones(9), {flatten_MxN(R)});


  // constrain T to identity for now
  //prog.AddLinearConstraint(Eigen::MatrixXd::Identity(3, 3),
  //  Eigen::VectorXd::Zero(3), Eigen::VectorXd::Zero(3), 
  //  {T});


  bool free_rot = true;
  if (free_rot){
    addMcCormickQuaternionConstraint(prog, R, 4, 4);
  } else {
    // constrain rotations to ground truth
    // I know I can do this in one constraint with 9 rows, but eigen was giving me trouble
    for (int i=0; i<3; i++){
      for (int j=0; j<3; j++){
        prog.AddLinearEqualityConstraint(Eigen::MatrixXd::Identity(1, 1), scene_model_tf.rotation()(i, j), {R.block<1,1>(i, j)});
      }
    }
  }

  // Every scene (tactile) point should correspond to exactly 1 model point
  Eigen::MatrixXd C1 = Eigen::MatrixXd::Ones(1, model_pts_tf->size());
  for (size_t k=0; k<scene_pts_tf->size(); k++){
    prog.AddLinearEqualityConstraint(C1, 1, {C.block(k, 0, 1, model_pts_tf->size()).transpose()});
  }

  Eigen::VectorXd M = VectorXd(3*model_pts_tf->size());
  for (int j=0; j<model_pts_tf->size(); j++){
    M(3*j+0) = model_pts_tf->at(j).x;
    M(3*j+1) = model_pts_tf->at(j).y;
    M(3*j+2) = model_pts_tf->at(j).z;
  }

  // Every model point can't correpond to every single scene point (but can correspond to all but one of them)
  Eigen::MatrixXd C2 = Eigen::MatrixXd::Ones(1, scene_pts_tf->size());
  for (size_t k=0; k<model_pts_tf->size(); k++){
    prog.AddLinearConstraint(C2, 0, scene_pts_tf->size() - 1, {C.block(0, k, scene_pts_tf->size(), 1)});
  }

  // I'm adding a dummy var constrained to zero to 
  // fill out the diagonals of C_i. Without this, I converge to
  // strange solutions...
  auto C_dummy = prog.AddContinuousVariables(1, "c_dummy_zero");
  prog.AddLinearEqualityConstraint(Eigen::MatrixXd::Ones(1, 1), Eigen::MatrixXd::Zero(1, 1), {C_dummy});
  auto B = Eigen::RowVectorXd(1, 3+1+3*model_pts_tf->size());    
  B.block<1, 1>(0, 3) = MatrixXd::Ones(1, 1); // T bias term
  B.block(0, 4, 1, model_pts_tf->size()*3) = -1.0 * M.transpose();

  printf("Starting to add correspondence costs... ");
  for (int i=0; i<scene_pts_tf->size(); i++){
    printf("=");

    //printf("Starting row cost %d...\n", i);
    double start_cost = getUnixTime();

    auto C_i = DecisionVariableMatrixX(3, 3*model_pts_tf->size());
  
    for (int j=0; j<model_pts_tf->size(); j++){
      C_i(1, 3*j) = C_dummy(0,0);
      C_i(2, 3*j) = C_dummy(0,0);
      C_i(0, 3*j+1) = C_dummy(0,0);
      C_i(2, 3*j+1) = C_dummy(0,0);
      C_i(0, 3*j+2) = C_dummy(0,0);
      C_i(1, 3*j+2) = C_dummy(0,0);

      C_i(0, 3*j) = C(i, j);
      C_i(1, 3*j+1) = C(i, j);
      C_i(2, 3*j+2) = C(i, j);
    }
    // B, containing the scene and model points and a translation bias term, is used in all three
    // cost terms (for the x, y, z components of the final error) 
    auto s_xyz = Eigen::Vector3d(scene_pts_tf->at(i).x, scene_pts_tf->at(i).y, scene_pts_tf->at(i).z);
    B.block<1, 3>(0, 0) = s_xyz.transpose(); // Multiples R

    // printf("Setting up quadratic cost for scene pt %d, adding %lu new constraints...\n", i, 3*(3+1+3*model_pts_tf->size()));
    // cout << " ********** start B " << endl;
    // cout << B << endl;
    // cout << " ********* end B " << endl;

    //printf("\tElapsed %f after writing C_i and finishing B\n", getUnixTime() - start_cost);
    auto BtB = B.transpose() * B;
    //printf("\tElapsed %f after computing B.'B\n", getUnixTime() - start_cost);

    // Quadratic cost and Quadratic Error Cost would do the same thing here, with x0 = b = zeros()
    // But quadratic error cost is hundreds of times slower, I think because it forces more matrix multiplies
    // with these big matrices!
    prog.AddQuadraticCost(BtB, Eigen::VectorXd::Zero(3+1+3*model_pts_tf->size()), 
      {R.block<1, 3>(0, 0).transpose(), 
       T.block<1,1>(0,0),
       C_i.block(0,0,1,3*model_pts_tf->size()).transpose()});
    prog.AddQuadraticCost(BtB, Eigen::VectorXd::Zero(3+1+3*model_pts_tf->size()), 
      {R.block<1, 3>(1, 0).transpose(), 
       T.block<1,1>(1,0),
       C_i.block(1,0,1,3*model_pts_tf->size()).transpose()});
    prog.AddQuadraticCost(BtB, Eigen::VectorXd::Zero(3+1+3*model_pts_tf->size()), 
      {R.block<1, 3>(2, 0).transpose(), 
       T.block<1,1>(2,0),
       C_i.block(2,0,1,3*model_pts_tf->size()).transpose()});

    //printf("\tElapsed %fs after setting up cost\n", getUnixTime() - start_cost);
  }

  double now = getUnixTime();

  GurobiSolver gurobi_solver;
  MosekSolver mosek_solver;

  prog.SetSolverOption("GUROBI", "OutputFlag", 1);
  prog.SetSolverOption("GUROBI", "LogToConsole", 1);
  prog.SetSolverOption("GUROBI", "LogFile", "loggg.gur");
  prog.SetSolverOption("GUROBI", "DisplayInterval", 5);
  prog.SetSolverOption("GUROBI", "TimeLimit", 1200.0);
//  prog.SetSolverOption("GUROBI", "MIPGap", 1E-12);
//  prog.SetSolverOption("GUROBI", "Heuristics", 0.25);
//  prog.SetSolverOption("GUROBI", "FeasRelaxBigM", 1E6);
//  prog.SetSolverOption("GUROBI", "Cutoff", 50.0);
// isn't doing anything... not invoking this tool right?
//  prog.SetSolverOption("GUROBI", "TuneJobs", 8);
//  prog.SetSolverOption("GUROBI", "TuneResults", 3);

  auto out = gurobi_solver.Solve(prog);
  string problem_string = "rigidtf";
  double elapsed = getUnixTime() - now;

  //prog.PrintSolution();

  printf("Transform:\n");
  printf("\tTranslation: %f, %f, %f\n", T(0, 0).value(), T(1, 0).value(), T(2, 0).value());
  printf("\tRotation:\n");
  printf("\t\t%f, %f, %f\n", R(0, 0).value(), R(0, 1).value(), R(0, 2).value());
  printf("\t\t%f, %f, %f\n", R(1, 0).value(), R(1, 1).value(), R(1, 2).value());
  printf("\t\t%f, %f, %f\n", R(2, 0).value(), R(2, 1).value(), R(2, 2).value());


  for (int k_s=0; k_s<scene_pts_tf->size(); k_s++){
    for (int k_m=0; k_m<model_pts_tf->size(); k_m++){
      if (C(k_s, k_m).value() >= 0.5){
        printf("Corresp s%d->m%d at dist %f\n", k_s, k_m, pointDistance(scene_pts_tf->at(k_s), model_pts_tf->at(k_m)));
        std::stringstream ss_line;
        ss_line << "raw_correspondence_line" << k_m << "-" << k_s;
        viewer.addLine<PointType, PointType> (model_pts_tf->at(k_m), scene_pts_tf->at(k_s), 255, 0, 255, ss_line.str ());
      }
    }
  }

  // and add ground truth
  for (int k_s=0; k_s<scene_pts_tf->size(); k_s++){
    int k = correspondences_gt[k_s];
    std::stringstream ss_line;
    ss_line << "gt_correspondence_line" << k << "-" << k_s;
    viewer.addLine<PointType, PointType> (model_pts_tf->at(k), scene_pts_tf->at(k_s), 0, 255, 0, ss_line.str ());
  }

  printf("Code %d, problem %s solved for %lu scene, %lu model solved in: %f\n", out, problem_string.c_str(), scene_pts_tf->size(), model_pts_tf->size(), elapsed);

  while (!viewer.wasStopped ())
    viewer.spinOnce ();

  return 0;
}
