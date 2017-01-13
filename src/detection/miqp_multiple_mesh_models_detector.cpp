/*
 */

#include <string>
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

const double kBigNumber = 100.0; // must be bigger than largest possible correspondance distance

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

struct Model {
  std::string name;
  Eigen::Affine3d scene_transform;
  Eigen::Affine3d model_transform;
  Eigen::Matrix3Xd vertices;
  std::vector< std::vector<int> > faces; // each face is a list of vertex indices, clockwise around the face
};
Model load_model_from_yaml_node(YAML::Node model_node){
  Model m;
  m.name = model_node["name"].as<string>();
  printf("Loading model %s...\n", m.name.c_str());
  int num_verts = model_node["vertices"].size();
  printf("\twith %d verts\n", num_verts);
  m.vertices.resize(3, num_verts);
  std::map<string, int> vertex_name_to_index;
  int k=0;
  for (auto iter = model_node["vertices"].begin(); iter != model_node["vertices"].end(); iter++){
    string name = iter->first.as<string>();
    vertex_name_to_index[name] = k;
    auto pt = iter->second.as<vector<double>>();
    for (int i=0; i<3; i++)
      m.vertices(i, k) = pt[i];
    k++;
  }

  printf("\twith %ld faces\n", model_node["faces"].size());
  for (auto iter = model_node["faces"].begin(); iter != model_node["faces"].end(); iter++){
    auto face_str = iter->second.as<vector<string>>();
    vector<int> face_int;
    for (int i=0; i<face_str.size(); i++){
      face_int.push_back(vertex_name_to_index[face_str[i]]);
    }
    m.faces.push_back(face_int);
  }

  vector<double> scene_tf = model_node["scene_tf"].as<vector<double>>();
  m.scene_transform.setIdentity();
  m.scene_transform.translation() = Vector3d(scene_tf[0], scene_tf[1], scene_tf[2]);
  // todo: ordering, or really just move to quat
  m.scene_transform.rotate (Eigen::AngleAxisd (scene_tf[5], Eigen::Vector3d::UnitZ()));
  m.scene_transform.rotate (Eigen::AngleAxisd (scene_tf[4], Eigen::Vector3d::UnitY()));
  m.scene_transform.rotate (Eigen::AngleAxisd (scene_tf[3], Eigen::Vector3d::UnitX()));

  vector<double> model_tf = model_node["model_tf"].as<vector<double>>();
  m.model_transform.setIdentity();
  m.model_transform.translation() = Vector3d(model_tf[0], model_tf[1], model_tf[2]);
  // todo: ordering, or really just move to quat
  m.model_transform.rotate (Eigen::AngleAxisd (model_tf[5], Eigen::Vector3d::UnitZ()));
  m.model_transform.rotate (Eigen::AngleAxisd (model_tf[4], Eigen::Vector3d::UnitY()));
  m.model_transform.rotate (Eigen::AngleAxisd (model_tf[3], Eigen::Vector3d::UnitX()));

  return m;
}
Eigen::Matrix3Xd get_face_midpoints(Model m){
  Matrix3Xd out(3, m.faces.size());
  for (int i=0; i<m.faces.size(); i++){
    double w0 = 0.3333;
    double w1 = 0.3333;
    double w2 = 0.3333;
    out.col(i) = w0*m.vertices.col(m.faces[i][0]) +  
                 w1*m.vertices.col(m.faces[i][1]) + 
                 w2*m.vertices.col(m.faces[i][2]);
  }
  return out;
}
Eigen::Matrix3Xd sample_from_surface_of_model(Model m, int N){
  Matrix3Xd out(3, N);
  for (int i=0; i<N; i++){
    int face = rand() % m.faces.size(); // assume < RAND_MAX faces...
    double w0 = randrange(0., 1.0);
    double w1 = randrange(0., 1.0);
    double w2 = randrange(0., 1.0);
    double tot = w0 + w1 + w2;
    w0 /= tot; w1 /= tot; w2 /= tot;
    out.col(i) = w0*m.vertices.col(m.faces[face][0]) +  
                 w1*m.vertices.col(m.faces[face][1]) + 
                 w2*m.vertices.col(m.faces[face][2]);
  }
  return out;
}

bool pending_redraw = true;
bool draw_all_mode = true;
bool reextract_solution = true;
int target_sol = 0;
int max_num_sols = 1;
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
  else if (event.getKeySym() == "Up" && event.keyDown()) {
    target_sol = (target_sol - 1 + max_num_sols) % max_num_sols;
    reextract_solution = true;
  }
  else if (event.getKeySym() == "Down" && event.keyDown()) {
    target_sol = (target_sol - 1 + max_num_sols) % max_num_sols;
    reextract_solution = true;
  }
  pending_redraw = true;
}

int main(int argc, char** argv) {
  srand(getUnixTime());

  int optNumRays = 10;
  int optRotationConstraint = 4;
  int optSceneSamplingMode = 0;

  if (argc != 2){
    printf("Use: miqp_multiple_mesh_models_detector <config file>\n");
    exit(-1);
  }

  std::shared_ptr<lcm::LCM> lcm(new lcm::LCM());
  if (!lcm->good()) {
    throw std::runtime_error("LCM is not good");
  }

  // Set up robot
  string yamlString = string(argv[1]);
  YAML::Node modelsNode = YAML::LoadFile(yamlString);

  if (modelsNode["options"]["num_rays"])
    optNumRays = modelsNode["options"]["num_rays"].as<int>();

  if (modelsNode["options"]["rotation_constraint"])
    optRotationConstraint = modelsNode["options"]["rotation_constraint"].as<int>();

  if (modelsNode["options"]["scene_sampling_mode"])
    optSceneSamplingMode = modelsNode["options"]["scene_sampling_mode"].as<int>();


  std::vector<Model> models;

  printf("Loaded\n");
  for (auto iter=modelsNode["models"].begin(); iter != modelsNode["models"].end(); iter++){
    models.push_back(load_model_from_yaml_node(*iter));
  }
  printf("Parsed models\n");

  // Render scene cloud by sampling surface of objects
  pcl::PointCloud<PointType>::Ptr scene_pts (new pcl::PointCloud<PointType> ());
  pcl::PointCloud<PointType>::Ptr actual_model_pts (new pcl::PointCloud<PointType> ());
  map<int, int> correspondences_gt;

  int k=0;
  for (auto iter = models.begin(); iter != models.end(); iter++){
    Matrix3Xd scene_samples;
    switch (optSceneSamplingMode){
      case 0:
        scene_samples = get_face_midpoints(*iter);
        break;
      case 1:
        scene_samples = sample_from_surface_of_model(*iter, optNumRays / models.size());
        break;
      default:
        printf("Bad scene sampling mode! %d\n", optSceneSamplingMode);
        exit(-1);
    }
    auto samples_model_frame = iter->model_transform * scene_samples;
    auto samples_scene_frame = iter->scene_transform * scene_samples;
    for (int i=0; i<samples_scene_frame.cols(); i++){
      scene_pts->push_back(PointType( samples_scene_frame(0, i), samples_scene_frame(1, i), samples_scene_frame(2, i)));
      // overkill doing this right now, as the scene = simpletf * model in same order. but down the road may get
      // more imaginative with this
      actual_model_pts->push_back(PointType( samples_model_frame(0, i), samples_model_frame(1, i), samples_model_frame(2, i)));
      correspondences_gt[k] = k;
      k++;
    }
  }

  printf("Running with %d scene pts\n", (int)scene_pts->size());


  // Do meshing conversions and setup -- models vertices all stored
  // at origin for now, overlapping each other
  // calculate total # of vertices
  int total_num_verts = 0;
  int total_num_faces = 0;
  for (int i=0; i<models.size(); i++){
    total_num_verts += models[i].vertices.cols();
    total_num_faces += models[i].faces.size();
  }

  Matrix3Xd vertices(3, total_num_verts);
  MatrixXd F(total_num_faces, total_num_verts);
  MatrixXd B(models.size(), total_num_faces);
  vertices.setZero();
  B.setZero();
  F.setZero();
  int verts_start = 0;
  int faces_start = 0;
  int face_ind = 0;
  for (int i=0; i<models.size(); i++){
    int num_verts = models[i].vertices.cols();
    int num_faces = models[i].faces.size();
    vertices.block(0, verts_start, 3, models[i].vertices.cols()) = models[i].model_transform * models[i].vertices;
    // Generate sub-block of face selection matrix F
    for (auto iter=models[i].faces.begin(); iter!=models[i].faces.end(); iter++){
      F(face_ind, verts_start+iter->at(0)) = 1.;
      F(face_ind, verts_start+iter->at(1)) = 1.;
      F(face_ind, verts_start+iter->at(2)) = 1.;
      face_ind++;
    }
    // Generate sub-block of object-to-face selection matrix B
    B.block(i, faces_start, 1, num_faces) = VectorXd::Ones(num_faces).transpose();
    verts_start += num_verts;
    faces_start += num_faces;
  }

  // See https://www.sharelatex.com/project/5850590c38884b7c6f6aedd1
  // for problem formulation
  MathematicalProgram prog;

  // Allocate slacks to choose minimum L-1 norm over objects
  auto phi = prog.NewContinuousVariables(scene_pts->size(), 1, "phi");
  
  // And slacks to store term-wise absolute value terms for L-1 norm calculation
  auto alpha = prog.NewContinuousVariables(3, scene_pts->size(), "alpha");

  // Each row is a set of affine coefficients relating the scene point to a combination
  // of vertices on a single face of the model
  auto C = prog.NewContinuousVariables(scene_pts->size(), vertices.cols(), "C");
  // Binary variable selects which face is being corresponded to
  auto f = prog.NewBinaryVariables(scene_pts->size(), F.rows(),"f");

  struct TransformationVars {
    DecisionVariableVectorX T;
    DecisionVariableMatrixX R;
  };
  std::vector<TransformationVars> transform_by_object;
  for (int i=0; i<models.size(); i++){
    TransformationVars new_tr;
    char name_postfix[100];
    sprintf(name_postfix, "_%s_%d", models[i].name.c_str(), i);
    new_tr.T = prog.NewContinuousVariables(3, string("T")+string(name_postfix));
    prog.AddBoundingBoxConstraint(-100*VectorXd::Ones(3), 100*VectorXd::Ones(3), {new_tr.T});
    new_tr.R = NewRotationMatrixVars(&prog, string("R") + string(name_postfix));

    if (optRotationConstraint > 0){
      switch (optRotationConstraint){
        case 1:
          break;
        case 2:
          // Columnwise and row-wise L1-norm >=1 constraints
          for (int k=0; k<3; k++){
            prog.AddLinearConstraint(Vector3d::Ones().transpose(), 1.0, std::numeric_limits<double>::infinity(), {new_tr.R.row(k).transpose()});
            prog.AddLinearConstraint(Vector3d::Ones().transpose(), 1.0, std::numeric_limits<double>::infinity(), {new_tr.R.col(k)});
          }
          break;
        case 3:
          addMcCormickQuaternionConstraint(prog, new_tr.R, 4, 4);
          break;
        case 4:
          AddRotationMatrixMcCormickEnvelopeMilpConstraints(&prog, new_tr.R);
          break;
        case 5:
          AddBoundingBoxConstraintsImpliedByRollPitchYawLimits(&prog, new_tr.R, kYaw_0_to_PI_2 | kPitch_0_to_PI_2 | kRoll_0_to_PI_2);
          break;
        default:
          printf("invalid optRotationConstraint option!\n");
          exit(-1);
          break;
      }
    } else {
      // constrain rotations to ground truth
      // I know I can do this in one constraint with 9 rows, but eigen was giving me trouble
      auto ground_truth_tf = models[i].scene_transform.inverse().cast<float>() * models[i].model_transform.cast<float>();
      for (int i=0; i<3; i++){
        for (int j=0; j<3; j++){
          prog.AddLinearEqualityConstraint(Eigen::MatrixXd::Identity(1, 1), ground_truth_tf.rotation()(i, j), {new_tr.R.block<1,1>(i, j)});
        }
      }
    }

    transform_by_object.push_back(new_tr);
  }

  // Optimization pushes on slacks to make them tight (make them do their job)
  prog.AddLinearCost(1.0 * VectorXd::Ones(scene_pts->size()), {phi});
  for (int k=0; k<3; k++){
    prog.AddLinearCost(1.0 * VectorXd::Ones(alpha.cols()), {alpha.row(k)});
  }

  // Constrain slacks nonnegative, to help the estimation of lower bound in relaxation  
  prog.AddBoundingBoxConstraint(0.0, std::numeric_limits<double>::infinity(), {phi});
  for (int k=0; k<3; k++){
    prog.AddBoundingBoxConstraint(0.0, std::numeric_limits<double>::infinity(), {alpha.row(k)});
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

  // I'm adding a dummy var constrained to zero to 
  // fill out the diagonals of C_i.
  auto C_dummy = prog.NewContinuousVariables(1, "c_dummy_zero");
  prog.AddLinearEqualityConstraint(Eigen::MatrixXd::Ones(1, 1), Eigen::MatrixXd::Zero(1, 1), {C_dummy});

  // Helper variable to produce linear constraint
  // alpha_{i, l} +/- (R_l * s_i + T - M C_{i, :}^T) - Big * B_l * f_i >= -Big
  auto AlphaConstrPos = Eigen::RowVectorXd(1, 1+3+1+vertices.cols() + B.cols());    
  AlphaConstrPos.block<1, 1>(0, 0) = MatrixXd::Ones(1, 1); // multiplies alpha_{i, l} elem
  AlphaConstrPos.block<1, 1>(0, 4) = -1.0 * MatrixXd::Ones(1, 1); // T bias term
  auto AlphaConstrNeg = Eigen::RowVectorXd(1, 1+3+1+vertices.cols() + B.cols());    
  AlphaConstrNeg.block<1, 1>(0, 0) = MatrixXd::Ones(1, 1); // multiplies alpha_{i, l} elem
  AlphaConstrNeg.block<1, 1>(0, 4) = MatrixXd::Ones(1, 1); // T bias term

  printf("Starting to add correspondence costs... ");
  for (int l=0; l<models.size(); l++){
    AlphaConstrPos.block(0, 5+vertices.cols(), 1, B.cols()) = -kBigNumber*B.row(l); // multiplies f_i
    AlphaConstrNeg.block(0, 5+vertices.cols(), 1, B.cols()) = -kBigNumber*B.row(l); // multiplies f_i

    for (int i=0; i<scene_pts->size(); i++){
      printf("=");

      // constrain L-1 distance slack based on correspondences
      // phi_i >= 1^T alpha_{i}
      // phi_i - 1&T alpha_{i} >= 0
      RowVectorXd PhiConstr(1 + 3);
      PhiConstr.setZero();
      PhiConstr(0, 0) = 1.0; // multiplies phi
      PhiConstr.block<1,3>(0,1) = -RowVector3d::Ones(); // multiplies alpha
      prog.AddLinearConstraint(PhiConstr, 0, std::numeric_limits<double>::infinity(),
      {phi.block<1,1>(i, 0),
       alpha.col(i)});


      // Alphaconstr, containing the scene and model points and a translation bias term, is used the constraints
      // on the three elems of alpha_{i, l}
      auto s_xyz = Eigen::Vector3d(scene_pts->at(i).x, scene_pts->at(i).y, scene_pts->at(i).z);
      AlphaConstrPos.block<1, 3>(0, 1) = -s_xyz.transpose(); // Multiples R
      AlphaConstrNeg.block<1, 3>(0, 1) = s_xyz.transpose(); // Multiples R

      AlphaConstrPos.block(0, 5, 1, vertices.cols()) = 1.0 * vertices.row(0); // multiplies the selection vars
      prog.AddLinearConstraint(AlphaConstrPos, -kBigNumber, std::numeric_limits<double>::infinity(),
        {alpha.block<1,1>(0, i),
         transform_by_object[l].R.block<1, 3>(0, 0).transpose(), 
         transform_by_object[l].T.block<1,1>(0,0),
         C.row(i).transpose(),
         f.row(i).transpose()});

      AlphaConstrPos.block(0, 5, 1, vertices.cols()) = 1.0 * vertices.row(1); // multiplies the selection vars
      prog.AddLinearConstraint(AlphaConstrPos, -kBigNumber, std::numeric_limits<double>::infinity(),
        {alpha.block<1,1>(1, i),
         transform_by_object[l].R.block<1, 3>(1, 0).transpose(), 
         transform_by_object[l].T.block<1,1>(1,0),
         C.row(i).transpose(),
         f.row(i).transpose()});

      AlphaConstrPos.block(0, 5, 1, vertices.cols()) = 1.0 * vertices.row(2); // multiplies the selection vars
      prog.AddLinearConstraint(AlphaConstrPos, -kBigNumber, std::numeric_limits<double>::infinity(),
        {alpha.block<1,1>(2, i),
         transform_by_object[l].R.block<1, 3>(2, 0).transpose(), 
         transform_by_object[l].T.block<1,1>(2,0),
         C.row(i).transpose(),
         f.row(i).transpose()});

      AlphaConstrNeg.block(0, 5, 1, vertices.cols()) = -1.0 * vertices.row(0); // multiplies the selection vars
      prog.AddLinearConstraint(AlphaConstrNeg, -kBigNumber, std::numeric_limits<double>::infinity(),
        {alpha.block<1,1>(0, i),
         transform_by_object[l].R.block<1, 3>(0, 0).transpose(), 
         transform_by_object[l].T.block<1,1>(0,0),
         C.row(i).transpose(),
         f.row(i).transpose()});
      AlphaConstrNeg.block(0, 5, 1, vertices.cols()) = -1.0 * vertices.row(1); // multiplies the selection vars
      prog.AddLinearConstraint(AlphaConstrNeg, -kBigNumber, std::numeric_limits<double>::infinity(),
        {alpha.block<1,1>(1, i),
         transform_by_object[l].R.block<1, 3>(1, 0).transpose(), 
         transform_by_object[l].T.block<1,1>(1,0),
         C.row(i).transpose(),
         f.row(i).transpose()});
      AlphaConstrNeg.block(0, 5, 1, vertices.cols()) = -1.0 * vertices.row(2); // multiplies the selection vars
      prog.AddLinearConstraint(AlphaConstrNeg, -kBigNumber, std::numeric_limits<double>::infinity(),
        {alpha.block<1,1>(2, i),
         transform_by_object[l].R.block<1, 3>(2, 0).transpose(), 
         transform_by_object[l].T.block<1,1>(2,0),
         C.row(i).transpose(),
         f.row(i).transpose()});
    }
  }
  printf("\n");

  double now = getUnixTime();

  GurobiSolver gurobi_solver;
  MosekSolver mosek_solver;

  prog.SetSolverOption("GUROBI", "OutputFlag", 1);
  prog.SetSolverOption("GUROBI", "LogToConsole", 1);
  prog.SetSolverOption("GUROBI", "LogFile", "loggg.gur");
  prog.SetSolverOption("GUROBI", "DisplayInterval", 5);

  if (modelsNode["options"]["gurobi_int_options"]){
    for (auto iter = modelsNode["options"]["gurobi_int_options"].begin();
         iter != modelsNode["options"]["gurobi_int_options"].end();
         iter++){
      prog.SetSolverOption("GUROBI", iter->first.as<string>(), iter->second.as<int>());
    }
  }
  if (modelsNode["options"]["gurobi_float_options"]){
    for (auto iter = modelsNode["options"]["gurobi_float_options"].begin();
         iter != modelsNode["options"]["gurobi_float_options"].end();
         iter++){
      prog.SetSolverOption("GUROBI", iter->first.as<string>(), iter->second.as<float>());
    }
  }

//  prog.SetSolverOption("GUROBI", "Cutoff", 50.0);
// isn't doing anything... not invoking this tool right?
//  prog.SetSolverOption("GUROBI", "TuneJobs", 8);
//  prog.SetSolverOption("GUROBI", "TuneResults", 3);
  //prog.SetSolverOption("GUROBI", )

  if (modelsNode["options"]["gurobi_int_options"]["PoolSolutions"])
    max_num_sols = modelsNode["options"]["gurobi_int_options"]["PoolSolutions"].as<int>();

  auto out = gurobi_solver.Solve(prog);
  string problem_string = "rigidtf";
  double elapsed = getUnixTime() - now;

  //prog.PrintSolution();
  printf("Code %d, problem %s solved for %lu scene solved in: %f\n", out, problem_string.c_str(), scene_pts->size(), elapsed);
  
  // Extract into a set of correspondences
  struct PointCorrespondence {
    PointType scene_pt;
    PointType model_pt;
    std::vector<PointType> model_verts;
    std::vector<double> vert_weights;

    // for reference into optim program
    int scene_ind;
    int face_ind;
    std::vector<int> vert_inds;
  };

  struct ObjectDetection {
    Eigen::Affine3f est_tf;
    std::vector<PointCorrespondence> correspondences;
    int obj_ind;
  };


  // Viewer main loop
  // Pressing left-right arrow keys allows viewing of individual point-face correspondences
  // Pressing "z" toggles viewing everything at once or doing individual-correspondence viewing
  // Pressing up-down arrow keys scrolls through different optimal solutions (TODO(gizatt) make this happen)

  pcl::visualization::PCLVisualizer viewer ("Point Collection");
  viewer.setShowFPS(false);
  pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_color_handler (scene_pts, 255, 255, 128);
  pcl::visualization::PointCloudColorHandlerCustom<PointType> model_color_handler (actual_model_pts, 255, 255, 128);
  viewer.registerKeyboardCallback (keyboardEventOccurred, (void*)&viewer);

  max_corresp_id = scene_pts->size();
  pending_redraw = true;


  std::vector<ObjectDetection> detections;
  MatrixXd f_est;
  MatrixXd C_est;
  reextract_solution = true;

  while (!viewer.wasStopped ()){
    if (reextract_solution){
      detections.clear();

      f_est= prog.GetSolution(f, target_sol);
      C_est = prog.GetSolution(C, target_sol);

      for (int i=0; i<models.size(); i++){
        ObjectDetection detection;
        detection.obj_ind = i;

        printf("************************************************\n");
        printf("Concerning model %d (%s):\n", i, models[i].name.c_str());
        printf("------------------------------------------------\n");
        printf("Ground truth TF: ");
        auto ground_truth_tf = models[i].scene_transform.inverse().cast<float>() 
                              * models[i].model_transform.cast<float>();
        cout << ground_truth_tf.translation().transpose() << endl;
        cout << ground_truth_tf.matrix().block<3,3>(0,0) << endl;
        printf("------------------------------------------------\n");
        Vector3f Tf = prog.GetSolution(transform_by_object[i].T, target_sol).cast<float>();
        Matrix3f Rf = prog.GetSolution(transform_by_object[i].R, target_sol).cast<float>();
        printf("Transform:\n");
        printf("\tTranslation: %f, %f, %f\n", Tf(0, 0), Tf(1, 0), Tf(2, 0));
        printf("\tRotation:\n");
        printf("\t\t%f, %f, %f\n", Rf(0, 0), Rf(0, 1), Rf(0, 2));
        printf("\t\t%f, %f, %f\n", Rf(1, 0), Rf(1, 1), Rf(1, 2));
        printf("\t\t%f, %f, %f\n", Rf(2, 0), Rf(2, 1), Rf(2, 2));
        printf("------------------------------------------------\n");
        printf("Sanity check rotation: R^T R = \n");
        MatrixXf RfTRf = Rf.transpose() * Rf;
        printf("\t\t%f, %f, %f\n", RfTRf(0, 0), RfTRf(0, 1), RfTRf(0, 2));
        printf("\t\t%f, %f, %f\n", RfTRf(1, 0), RfTRf(1, 1), RfTRf(1, 2));
        printf("\t\t%f, %f, %f\n", RfTRf(2, 0), RfTRf(2, 1), RfTRf(2, 2));
        printf("------------------------------------------------\n");
        printf("************************************************\n");


        detection.est_tf.setIdentity();
        detection.est_tf.translation() = Tf;
        detection.est_tf.matrix().block<3,3>(0,0) = Rf;

        for (int scene_i=0; scene_i<f_est.rows(); scene_i++){
          for (int face_j=0; face_j<f_est.cols(); face_j++){
            // if this face is assigned, and this face is a member of this object,
            // then display this point
            if (f_est(scene_i, face_j) > 0.5 && B(i, face_j) > 0.5){
              PointCorrespondence new_corresp;
              new_corresp.scene_pt = scene_pts->at(scene_i);
              new_corresp.model_pt = transformPoint(scene_pts->at(scene_i), detection.est_tf);
              new_corresp.scene_ind = scene_i;
              new_corresp.face_ind = face_j;
              for (int k_v=0; k_v<vertices.cols(); k_v++){
                if (C_est(scene_i, k_v) >= 0.0){
                  new_corresp.model_verts.push_back( PointType(vertices(0, k_v), vertices(1, k_v), vertices(2, k_v)) );
                  new_corresp.vert_weights.push_back( C_est(target_corresp_id, k_v) );
                  new_corresp.vert_inds.push_back(k_v);
                }
              }
              detection.correspondences.push_back(new_corresp);
            }
          }
        }
        detections.push_back(detection);
      }
    }

    if (pending_redraw){
      viewer.removeAllPointClouds();
      viewer.removeAllShapes();

      viewer.addPointCloud<PointType>(scene_pts, scene_color_handler, "scene pts"); 
      viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "scene pts");
      viewer.addPointCloud<PointType>(actual_model_pts, model_color_handler, "model pts gt"); 
      viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "model pts gt");
 
      if (draw_all_mode){
        // Draw all correspondneces as lines
        for (auto det = detections.begin(); det != detections.end(); det++){
          for (auto corr = det->correspondences.begin(); corr != det->correspondences.end(); corr++){
            std::stringstream ss_line;
            ss_line << "raw_correspondence_line" << det->obj_ind << "-" << corr->face_ind << "-" << corr->scene_ind;
            viewer.addLine<PointType, PointType> (corr->scene_pt, corr->model_pt, 255, 0, 255, ss_line.str ());
          }
        }

        // and add all ground truth
        for (int k_s=0; k_s<scene_pts->size(); k_s++){
          std::stringstream ss_line;
          int k = correspondences_gt[k_s];
          ss_line << "gt_correspondence_line" << k << "-" << k_s;
          viewer.addLine<PointType, PointType> (actual_model_pts->at(k), scene_pts->at(k_s), 0, 255, 0, ss_line.str ());
        }
      } else {
        // Draw only desired correspondence
        for (auto det = detections.begin(); det != detections.end(); det++){
          for (auto corr = det->correspondences.begin(); corr != det->correspondences.end(); corr++){
            if (corr->scene_ind == target_corresp_id){
              std::stringstream ss_line;
              ss_line << "raw_correspondence_line" << det->obj_ind << "-" << corr->face_ind << "-" << corr->scene_ind;
              viewer.addLine<PointType, PointType> (corr->scene_pt, corr->model_pt, 255, 0, 255, ss_line.str ());
            }
          }
        }

        // And desired ground truth
        int k = correspondences_gt[target_corresp_id];
        std::stringstream ss_line;
        ss_line << "gt_correspondence_line" << k << "-" << target_corresp_id;
        viewer.addLine<PointType, PointType> (actual_model_pts->at(k), scene_pts->at(target_corresp_id), 0, 255, 0, ss_line.str ());
        // Re-draw the corresponded vertices larger
        for (int k_v=0; k_v<vertices.cols(); k_v++){
          if (prog.GetSolution(C(target_corresp_id, k_v), target_sol) >= 0.0){
            std::stringstream ss_sphere;
            ss_sphere << "vertex_sphere" << k_v;
            viewer.addSphere<PointType>(PointType(vertices(0, k_v), vertices(1, k_v), vertices(2, k_v)), prog.GetSolution(C(target_corresp_id, k_v), target_sol)/100.0, 
              255,0,255, ss_sphere.str());
          }
        }

      }

      // Always draw the transformed and untransformed models
      for (int i=0; i<models.size(); i++){
        auto verts = models[i].vertices;
        for (int j=0; j<models[i].faces.size(); j++){
          pcl::PointCloud<PointType>::Ptr face_pts (new pcl::PointCloud<PointType> ());
          for (int k=0; k<3; k++){
            face_pts->push_back(
              PointType( verts(0, models[i].faces[j][k]),
                         verts(1, models[i].faces[j][k]),
                         verts(2, models[i].faces[j][k]) ));
          } 
          char strname[100];
          // model pts
          sprintf(strname, "polygon%d_%d", i, j);
          transformPointCloud(*face_pts, *face_pts, models[i].model_transform.cast<float>());
          viewer.addPolygon<PointType>(face_pts, 0.5, 0.5, 1.0, string(strname));
          transformPointCloud(*face_pts, *face_pts, models[i].model_transform.inverse().cast<float>());

          // scene pts
          sprintf(strname, "polygon%d_%d_tf", i, j);
          transformPointCloud(*face_pts, *face_pts, models[i].scene_transform.cast<float>());
          viewer.addPolygon<PointType>(face_pts, 0.5, 1.0, 0.5, string(strname));
        }
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
