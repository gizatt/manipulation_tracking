/*
 */

#include <stdexcept>
#include <iostream>

#include "drake/multibody/rigid_body_tree.h"
#include "drake/multibody/parsers/urdf_parser.h"
#include "drake/solvers/mathematical_program.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/rotation_constraint.h"
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
using namespace drake::parsers::urdf;

typedef pcl::PointXYZ PointType;
typedef pcl::Normal NormalType;

MatrixXd boundingBox2FaceSel(Matrix3Xd bb_pts){

  // Get average per axis, which we'll use to distinguish
  // the postive vs negative face on each axis
  Vector3d avg;
  avg.setZero();
  for (int k=0; k<bb_pts.cols(); k++){
    avg += bb_pts.col(k);
  }
  avg /= (double)bb_pts.cols();

  // order in:
  // cx << -1, 1, 1, -1, -1, 1, 1, -1;
  // cy << 1, 1, 1, 1, -1, -1, -1, -1;
  // cz << 1, 1, -1, -1, -1, -1, 1, 1;
  MatrixXd F(6, bb_pts.cols());
  F.setZero();

  vector<vector<Vector3d>> out;
  int k=0;
  for (int xyz=0; xyz<3; xyz+=1){
    for (int tar=-1; tar<=1; tar+=2){
      for (int i=0; i<8; i++){
        if ((bb_pts(xyz, i)-avg(xyz))*(double)tar >= 0){
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


bool pending_redraw = true;
bool draw_all_mode = true;
int target_corresp_id = 0;
int max_corresp_id = 0;
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
                            void* viewer_void)
{
  pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
  if (event.getKeySym () == "z" && event.keyDown ())
  { 
    draw_all_mode = !draw_all_mode;
  }
  else if (event.getKeySym() == "Right" && event.keyDown()) {
    target_corresp_id = (target_corresp_id + 1) % max_corresp_id;
  }
  else if (event.getKeySym() == "Left" && event.keyDown()) {
    target_corresp_id = (target_corresp_id - 1 + max_corresp_id) % max_corresp_id;
  }
  pending_redraw = true;
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
  printf("getting in\n");
  string urdfString = string(argv[3]);
  RigidBodyTree<double> robot;
  AddModelInstanceFromUrdfFileWithRpyJointToWorld(urdfString, &robot);
  VectorXd q0_robot(robot.get_num_positions());
  KinematicsCache<double> robot_kinematics_cache = robot.doKinematics(q0_robot);
  printf("Set up robot with %d positions\n", robot.get_num_positions());
  printf("here after kin\n");
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
  //scene_centering_tf.translation() = avg_scene_pt.cast<float>();
  pcl::transformPointCloud (*scene_pts, *scene_pts_tf, scene_centering_tf);

  // Randomly transform model point cloud
  Eigen::Affine3f scene_model_tf = Eigen::Affine3f::Identity();
  scene_model_tf.translation() << randrange(-0.5, 0.5), randrange(-0.5, 0.5), randrange(-0.5, 0.5);
  // theta radians arround Z axis
  scene_model_tf.rotate (Eigen::AngleAxisf (randrange(0, 1.57), Eigen::Vector3f::UnitZ()));

  pcl::PointCloud<PointType>::Ptr model_pts_tf (new pcl::PointCloud<PointType> ());
  pcl::transformPointCloud (*model_pts, *model_pts_tf, scene_model_tf);

  printf("Selected %ld model pts and %ld scene pts\n", model_pts->size(), scene_pts_tf->size());

  // Do meshing conversions and setup
  Matrix3Xd vertices;
  robot.get_body(1).get_visual_elements()[0].getGeometry().getPoints(vertices);
  vertices = scene_model_tf.cast<double>() * vertices;
  // Generate face selection matrix F
  MatrixXd F = boundingBox2FaceSel(vertices);


  // We want to solve
  //  \min \sum_{i \in N_s} ||R s_i + T - C_i V||^2
  // R being a 3x3 rotation matrix, T being a 3x1 translation, 
  // C_i being a collection of affine correspondence variables
  // relating scene point i to some model point, and V being
  // the set of model vertices
  // V = {3*N_v, 1} matrix appending all model vertices
  // C_i = {3 x 3*N_v} matrix with structure
  //  [ c_{i, 1} * I(3) ... ... ... c_{i, N_m} * I(3) ]
  // 
  // To get things in the right order, we instead minimize
  // \min \sum_{i \in N_s} ||s_i^T R^T + T^T - M^T C_i^T||^2
  // Because || B * x ||^2 = x^T B^T B x
  // and x = [R^T
  //          T^T
  //          C_i^T]
  // B = [s_i^T   1  -V^T]
  // giving us our quadratic cost

  MathematicalProgram prog;

  // Each row is a set of affine coefficients relating the scene point to a combination
  // of vertices on a single face of the model
  auto C = prog.NewContinuousVariables(scene_pts_tf->size(), vertices.cols(), "C");
  // Binary variable selects which face is being corresponded to
  auto f = prog.NewBinaryVariables(scene_pts_tf->size(), F.rows(),"f");

  auto T = prog.NewContinuousVariables(3, 1, "T");
  // And add bounding box constraints on appropriate variables
  prog.AddBoundingBoxConstraint(-100*VectorXd::Ones(3), 100*VectorXd::Ones(3), {flatten_MxN(T)});

  auto R = NewRotationMatrixVars(&prog, "R");

  // constrain T to identity for now
  //prog.AddLinearConstraint(Eigen::MatrixXd::Identity(3, 3),
  //  Eigen::VectorXd::Zero(3), Eigen::VectorXd::Zero(3), 
  //  {T});

  bool free_rot = true;
  if (free_rot){
    //addMcCormickQuaternionConstraint(prog, R, 4, 4);
    //AddBoundingBoxConstraintsImpliedByRollPitchYawLimits(&prog, R, kYaw_0_to_PI_2 | kPitch_0_to_PI_2 | kRoll_0_to_PI_2);
    AddRotationMatrixMcCormickEnvelopeMilpConstraints(&prog, R);
  } else {
    // constrain rotations to ground truth
    // I know I can do this in one constraint with 9 rows, but eigen was giving me trouble
    for (int i=0; i<3; i++){
      for (int j=0; j<3; j++){
        prog.AddLinearEqualityConstraint(Eigen::MatrixXd::Identity(1, 1), scene_model_tf.rotation()(i, j), {R.block<1,1>(i, j)});
      }
    }
  }

  // Constrain each row of C to sum to 1, to make them proper
  // affine coefficients
  Eigen::MatrixXd C1 = Eigen::MatrixXd::Ones(1, C.cols());
  for (size_t k=0; k<C.rows(); k++){
    prog.AddLinearEqualityConstraint(C1, 1, {C.row(k).transpose()});
  }

  // Constrain each row of f to sum to 1, to force selection of exactly
  // one face to correspond to
  Eigen::MatrixXd f1 = Eigen::MatrixXd::Ones(1, f.cols());
  for (size_t k=0; k<f.rows(); k++){
    prog.AddLinearEqualityConstraint(f1, 1, {f.row(k).transpose()});
  }

  // Force all elems of C nonnegative
  for (int i=0; i<C.rows(); i++){
    for (int j=0; j<C.cols(); j++){
      prog.AddBoundingBoxConstraint(0.0, 1.0, C(i, j));
    }
  }

  // Force elems of C to be zero unless their corresponding vertex is a member
  // of an active face
  // That is,
  //   C_{i, j} <= F_{:, j}^T * f_{i, :}^T
  // or reorganized
  // [0] <= [F_{:, j}^T -1] [f_{i, :}^T C_{i, j}]
  //         
  for (int i=0; i<C.rows(); i++){
    for (int j=0; j<C.cols(); j++){
      MatrixXd A(1, F.rows() + 1);
      A.block(0, 0, 1, F.rows()) = F.col(j).transpose();
      A(0, F.rows()) = -1.0;

      prog.AddLinearConstraint(A, 0.0, std::numeric_limits<double>::infinity(), {f.row(i).transpose(), C.block<1,1>(i,j)});
    }
  }

  // flatten should be able to replace this but I'm having strange compile bugs
  // I don't want to fight through yet
  Eigen::VectorXd V = VectorXd(3*vertices.cols());
  for (int j=0; j<vertices.cols(); j++){
    V.block<3, 1>(3*j, 0) = vertices.col(j);
  }

  // I'm adding a dummy var constrained to zero to 
  // fill out the diagonals of C_i.
  auto C_dummy = prog.NewContinuousVariables(1, "c_dummy_zero");
  prog.AddLinearEqualityConstraint(Eigen::MatrixXd::Ones(1, 1), Eigen::MatrixXd::Zero(1, 1), {C_dummy});
  auto B = Eigen::RowVectorXd(1, 3+1+3*vertices.cols());    
  B.block<1, 1>(0, 3) = MatrixXd::Ones(1, 1); // T bias term
  B.block(0, 4, 1, vertices.cols()*3) = -1.0 * V.transpose();
  printf("c\n");

  printf("Starting to add correspondence costs... ");
  for (int i=0; i<scene_pts_tf->size(); i++){
    printf("=");

    //printf("Starting row cost %d...\n", i);
    double start_cost = getUnixTime();

    auto C_i = DecisionVariableMatrixX(3, 3*vertices.cols());
  
    for (int j=0; j<vertices.cols(); j++){
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

    auto BtB = B.transpose() * B;

    // Quadratic cost and Quadratic Error Cost would do the same thing here, with x0 = b = zeros()
    // But quadratic error cost is hundreds of times slower, I think because it forces more matrix multiplies
    // with these big matrices!
    prog.AddQuadraticCost(BtB, Eigen::VectorXd::Zero(3+1+3*vertices.cols()), 
      {R.block<1, 3>(0, 0).transpose(), 
       T.block<1,1>(0,0),
       C_i.block(0,0,1,3*vertices.cols()).transpose()});
    prog.AddQuadraticCost(BtB, Eigen::VectorXd::Zero(3+1+3*vertices.cols()), 
      {R.block<1, 3>(1, 0).transpose(), 
       T.block<1,1>(1,0),
       C_i.block(1,0,1,3*vertices.cols()).transpose()});
    prog.AddQuadraticCost(BtB, Eigen::VectorXd::Zero(3+1+3*vertices.cols()), 
      {R.block<1, 3>(2, 0).transpose(), 
       T.block<1,1>(2,0),
       C_i.block(2,0,1,3*vertices.cols()).transpose()});

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

  //prog.PrintSolution();o

  printf("*******\n");
  printf("Ground truth TF: ");
  cout << scene_model_tf.translation().transpose() << endl;
  cout << scene_model_tf.matrix().block<3,3>(0,0) << endl;
  printf("*******\n");


  Vector3f Tf = prog.GetSolution(T).cast<float>();
  Matrix3f Rf = prog.GetSolution(R).cast<float>();
  printf("Transform:\n");
  printf("\tTranslation: %f, %f, %f\n", Tf(0, 0), Tf(1, 0), Tf(2, 0));
  printf("\tRotation:\n");
  printf("\t\t%f, %f, %f\n", Rf(0, 0), Rf(0, 1), Rf(0, 2));
  printf("\t\t%f, %f, %f\n", Rf(1, 0), Rf(1, 1), Rf(1, 2));
  printf("\t\t%f, %f, %f\n", Rf(2, 0), Rf(2, 1), Rf(2, 2));
  printf("*******\n");

  printf("Sanity check rotation: R^T R = \n");
  MatrixXf RfTRf = Rf.transpose() * Rf;
  printf("\t\t%f, %f, %f\n", RfTRf(0, 0), RfTRf(0, 1), RfTRf(0, 2));
  printf("\t\t%f, %f, %f\n", RfTRf(1, 0), RfTRf(1, 1), RfTRf(1, 2));
  printf("\t\t%f, %f, %f\n", RfTRf(2, 0), RfTRf(2, 1), RfTRf(2, 2));
  printf("*******\n");


  Eigen::Affine3f est_tf = Eigen::Affine3f::Identity();
  est_tf.translation() = Tf;
  est_tf.matrix().block<3,3>(0,0) = Rf;

  pcl::PointCloud<PointType>::Ptr scene_pts_tf_est (new pcl::PointCloud<PointType> ());
  pcl::transformPointCloud (*scene_pts_tf, *scene_pts_tf_est, est_tf);

  printf("Code %d, problem %s solved for %lu scene, %lu model solved in: %f\n", out, problem_string.c_str(), scene_pts_tf->size(), model_pts_tf->size(), elapsed);



  // Viewer main loop
  // Pressing left-right arrow keys allows viewing of individual point-face correspondences
  // Pressing "z" toggles viewing everything at once or doing individual-correspondence viewing
  // Pressing up-down arrow keys scrolls through different optimal solutions (TODO(gizatt) make this happen)

  pcl::visualization::PCLVisualizer viewer ("Point Collection");
  viewer.setShowFPS(false);
  pcl::visualization::PointCloudColorHandlerCustom<PointType> model_color_handler (model_pts_tf, 128, 255, 255);
  pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_color_handler (scene_pts_tf, 255, 255, 128);
  viewer.registerKeyboardCallback (keyboardEventOccurred, (void*)&viewer);

  max_corresp_id = scene_pts_tf->size();
  pending_redraw = true;

  while (!viewer.wasStopped ()){
    if (pending_redraw){
      viewer.removeAllPointClouds();
      viewer.removeAllShapes();

      viewer.addPointCloud<PointType>(model_pts_tf, model_color_handler, "model pts_tf"); 
      viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "model pts_tf");
      viewer.addPointCloud<PointType>(scene_pts_tf, scene_color_handler, "scene pts tf"); 
      viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "scene pts tf");
 
      if (draw_all_mode){
        // Draw all correspondneces as lines
        for (int k_s=0; k_s<scene_pts_tf->size(); k_s++){
          for (int k_f=0; k_f<f.cols(); k_f++){
            if (prog.GetSolution(f(k_s, k_f)) >= 0.5){
              std::stringstream ss_line;
              ss_line << "raw_correspondence_line" << k_f << "-" << k_s;
              viewer.addLine<PointType, PointType> (scene_pts_tf_est->at(k_s), scene_pts_tf->at(k_s), 255, 0, 255, ss_line.str ());
            }
          }
        }
        // and add all ground truth
        for (int k_s=0; k_s<scene_pts_tf->size(); k_s++){
          int k = correspondences_gt[k_s];
          std::stringstream ss_line;
          ss_line << "gt_correspondence_line" << k << "-" << k_s;
          viewer.addLine<PointType, PointType> (model_pts_tf->at(k), scene_pts_tf->at(k_s), 0, 255, 0, ss_line.str ());
        }
      } else {
        // Draw only desired correspondence
        for (int k_f=0; k_f<f.cols(); k_f++){
          if (prog.GetSolution(f(target_corresp_id, k_f)) >= 0.5){
            std::stringstream ss_line;
            ss_line << "raw_correspondence_line" << k_f << "-" << target_corresp_id;
            viewer.addLine<PointType, PointType> (scene_pts_tf_est->at(target_corresp_id), scene_pts_tf->at(target_corresp_id), 255, 0, 255, ss_line.str ());
          }
        }
        // And desired ground truth
        int k = correspondences_gt[target_corresp_id];
        std::stringstream ss_line;
        ss_line << "gt_correspondence_line" << k << "-" << target_corresp_id;
        viewer.addLine<PointType, PointType> (model_pts_tf->at(k), scene_pts_tf->at(target_corresp_id), 0, 255, 0, ss_line.str ());
        // Re-draw the corresponded vertices larger
        for (int k_v=0; k_v<vertices.cols(); k_v++){
          if (prog.GetSolution(C(target_corresp_id, k_v)) >= 0.0){
            std::stringstream ss_sphere;
            ss_sphere << "vertex_sphere" << k_v;
            viewer.addSphere<PointType>(PointType(vertices(0, k_v), vertices(1, k_v), vertices(2, k_v)), prog.GetSolution(C(target_corresp_id, k_v))/100.0, 
              255,0,255, ss_sphere.str());
          }
        }

      }

      // Always draw the model mesh
      for (int i=0; i<F.rows(); i++){
        pcl::PointCloud<PointType>::Ptr face_pts (new pcl::PointCloud<PointType> ());
        for (int j=0; j<F.cols(); j++){
          if (F(i, j) > 0){
            VectorXd pt = vertices.col(j);
            face_pts->push_back(PointType(pt[0], pt[1], pt[2]));
          }
        } 
        char strname[100];
        sprintf(strname, "polygon%d", i);
        viewer.addPolygon<PointType>(face_pts, 0.2, 0.2, 1.0, string(strname));
      }

      // Always draw info
      std::stringstream ss_info;
      ss_info << "MIQP Scene Point to Model Mesh Correspondences" << endl
              << "Solution Objective: " << "todo" << endl
              << "Drawing mode [Z]: " << draw_all_mode << endl
              << "Drawing correspondence [Left/Right Keys]: " << target_corresp_id << endl;
      viewer.addText  ( ss_info.str(), 10, 10, "optim_info_str");
      

      pending_redraw = false;
    }

    viewer.spinOnce ();
  }

  return 0;
}
