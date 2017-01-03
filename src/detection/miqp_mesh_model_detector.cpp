/*
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

MatrixXd boundingBox2FaceSel(MatrixXd bb_pts){
  // order in:
  // cx << -1, 1, 1, -1, -1, 1, 1, -1;
  // cy << 1, 1, 1, 1, -1, -1, -1, -1;
  // cz << 1, 1, -1, -1, -1, -1, 1, 1;
  MatrixXd F(6, bb_pts.size());
  F.setZero();

  vector<vector<Vector3d>> out;
  int k=0;
  for (int xyz=0; xyz<3; xyz+=1){
    for (int tar=-1; tar<=1; tar+=2){
      for (int i=0; i<8; i++){
        if (bb_pts(xyz, i)*(double)tar >= 0){
          F(k, i) = 1.0;
        }
      }
      k++;
    }
  }
  return F;
}

vector< vector<Vector3d> > boundingBox2QuadMesh(MatrixXd bb_pts){
  // order in:
  // cx << -1, 1, 1, -1, -1, 1, 1, -1;
  // cy << 1, 1, 1, 1, -1, -1, -1, -1;
  // cz << 1, 1, -1, -1, -1, -1, 1, 1;
  vector<vector<Vector3d>> out;
  for (int xyz=0; xyz<3; xyz+=1){
    for (int tar=-1; tar<=1; tar+=2){
      vector<Vector3d> face;
      for (int i=0; i<8; i++){
        if (bb_pts(xyz, i)*(double)tar >= 0){
          face.push_back(bb_pts.col(i));
        }
      }
      out.push_back(face);
    }
  }
  return out;
}

int main(int argc, char** argv) {
  srand(getUnixTime());

  if (argc != 4){
    printf("Use: miqp_mesh_model_detector <numrays> <neighborhood size> <urdf>\n");
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

  // Do meshing conversions and setup
  Matrix3Xd vertices;
  robot.get_body(1).get_visual_elements()[0].getGeometry().getPoints(vertices);
  // Generate face selection matrix F
  MatrixXd F = boundingBox2FaceSel(vertices);
  // Draw the model mesh
  for (int i=0; i<F.rows(); i++){
    pcl::PointCloud<PointType>::Ptr face_pts (new pcl::PointCloud<PointType> ());
    printf("Mesh face %d:\n", i);
    for (int j=0; j<F.cols(); j++){
      if (F(i, j) > 0){
        cout << "\t" << vertices.col(j).transpose() << endl;
        VectorXd pt = scene_model_tf.cast<double>() * vertices.col(j);
        face_pts->push_back(PointType(pt[0], pt[1], pt[2]));
      }
    } 
    char strname[100];
    sprintf(strname, "polygon%d", i);
    viewer.addPolygon<PointType>(face_pts, 0.2, 0.2, 1.0, string(strname));
  }


  // Solve rigid body transform problem with correspondences
  MathematicalProgram prog;

  auto T = prog.AddContinuousVariables(3, 1, "t");
  // And add bounding box constraints on appropriate variables
  prog.AddBoundingBoxConstraint(-100*VectorXd::Ones(3), 100*VectorXd::Ones(3), {flatten_MxN(T)});

  auto R = prog.AddContinuousVariables(3, 3, "r");
  prog.AddBoundingBoxConstraint(-VectorXd::Ones(9), VectorXd::Ones(9), {flatten_MxN(R)});


  // constrain T to identity for now
  //prog.AddLinearConstraint(Eigen::MatrixXd::Identity(3, 3),
  //  Eigen::VectorXd::Zero(3), Eigen::VectorXd::Zero(3), 
  //  {T});


  bool free_rot = false;
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

  printf("Code %d, problem %s solved for %lu scene, %lu model solved in: %f\n", out, problem_string.c_str(), scene_pts_tf->size(), model_pts_tf->size(), elapsed);

  while (!viewer.wasStopped ())
    viewer.spinOnce ();

  return 0;
}
