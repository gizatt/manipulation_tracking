#include <stdexcept>
#include <iostream>

#include "drake/solvers/mathematical_program.h"
#include "drake/common/eigen_matrix_compare.h"
#include "drake/common/eigen_types.h"

#include <lcm/lcm-cpp.hpp>

#include "yaml-cpp/yaml.h"
#include "common/common.hpp"
#include "unistd.h"

#include "optimization_helpers.h"

#include <random>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/common/geometry.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>

using namespace std;
using namespace Eigen;
using namespace drake::solvers;

void testMcCormickApproxForward(double xmin, double xmax, double ymin, double ymax,
  int M_x, int M_y, double query_x, double query_y){
  MathematicalProgram prog;
  auto vars = prog.NewContinuousVariables(3, 1, {"xy", "x", "y"});
  add_McCormick_envelope(prog, vars(0, 0), vars(1, 0), vars(2, 0), string("mccorm"),
                         xmin, xmax,
                         ymin, ymax,
                         M_x, M_y);

  prog.AddQuadraticErrorCost(MatrixXd::Identity(2, 2), Vector2d(query_x, query_y), {vars.block<2, 1>(1, 0)});
  double now = getUnixTime();
  auto out = prog.Solve();
  string problem_string = "testmccormenv";
  double elapsed = getUnixTime() - now;
  printf("FWD: RET %02d, VARS %05lu, [%02.2f,%02.2f]/%02dx[%02.2f,%02.2f]/%02d->Query %02.2fx%02.2f->z=%02.2f\n",
    out, prog.num_vars(), xmin, xmax, M_x, ymin, ymax, M_y, query_x, query_y, prog.GetSolution(vars(0,0)));
/*
  printf("Sol: *********\n");
  prog.PrintSolution();
  printf("************\n");
*/
}

int main(int argc, char** argv) {
  srand(getUnixTime());

  if (argc != 1){
    printf("Use: test3DRotationApprox <noargs>\n");
    exit(-1);
  }

  std::shared_ptr<lcm::LCM> lcm(new lcm::LCM());
  if (!lcm->good()) {
    throw std::runtime_error("LCM is not good");
  }

  printf("Bound [0->1] tests:\n");
  printf("1x1 tests:\n");
  testMcCormickApproxForward(0, 1, 0, 1, 1, 1, 0.0, 0.0);
  testMcCormickApproxForward(0, 1, 0, 1, 1, 1, 0.0, 1.0);
  testMcCormickApproxForward(0, 1, 0, 1, 1, 1, 1.0, 1.0);
  testMcCormickApproxForward(0, 1, 0, 1, 1, 1, 0.0, 0.33);
  testMcCormickApproxForward(0, 1, 0, 1, 1, 1, 0.33, 0.0);
  testMcCormickApproxForward(0, 1, 0, 1, 1, 1, 0.33, 0.33);
  testMcCormickApproxForward(0, 1, 0, 1, 1, 1, 0.5, 0.5);
  testMcCormickApproxForward(0, 1, 0, 1, 1, 1, 0.5, 0.66);
  testMcCormickApproxForward(0, 1, 0, 1, 1, 1, 0.5, 0.5);

  printf("2x2 tests:\n");
  testMcCormickApproxForward(0, 1, 0, 1, 2, 2, 0.0, 0.0);
  testMcCormickApproxForward(0, 1, 0, 1, 2, 2, 0.0, 1.0);
  testMcCormickApproxForward(0, 1, 0, 1, 2, 2, 1.0, 1.0);
  testMcCormickApproxForward(0, 1, 0, 1, 2, 2, 0.0, 0.33);
  testMcCormickApproxForward(0, 1, 0, 1, 2, 2, 0.33, 0.0);
  testMcCormickApproxForward(0, 1, 0, 1, 2, 2, 0.33, 0.33);
  testMcCormickApproxForward(0, 1, 0, 1, 2, 2, 0.5, 0.5);
  testMcCormickApproxForward(0, 1, 0, 1, 2, 2, 0.5, 0.66);
  testMcCormickApproxForward(0, 1, 0, 1, 2, 2, 0.5, 0.5);

  printf("4x4 tests:\n");  
  testMcCormickApproxForward(0, 1, 0, 1, 4, 4, 0.0, 0.0);
  testMcCormickApproxForward(0, 1, 0, 1, 4, 4, 0.0, 1.0);
  testMcCormickApproxForward(0, 1, 0, 1, 4, 4, 1.0, 1.0);
  testMcCormickApproxForward(0, 1, 0, 1, 4, 4, 0.0, 0.33);
  testMcCormickApproxForward(0, 1, 0, 1, 4, 4, 0.33, 0.0);
  testMcCormickApproxForward(0, 1, 0, 1, 4, 4, 0.33, 0.33);
  testMcCormickApproxForward(0, 1, 0, 1, 4, 4, 0.5, 0.5);
  testMcCormickApproxForward(0, 1, 0, 1, 4, 4, 0.5, 0.66);
  testMcCormickApproxForward(0, 1, 0, 1, 4, 4, 0.5, 0.5);


  printf("8x8 tests:\n");  
  testMcCormickApproxForward(0, 1, 0, 1, 8, 8, 0.0, 0.0);
  testMcCormickApproxForward(0, 1, 0, 1, 8, 8, 0.0, 1.0);
  testMcCormickApproxForward(0, 1, 0, 1, 8, 8, 1.0, 1.0);
  testMcCormickApproxForward(0, 1, 0, 1, 8, 8, 0.0, 0.33);
  testMcCormickApproxForward(0, 1, 0, 1, 8, 8, 0.33, 0.0);
  testMcCormickApproxForward(0, 1, 0, 1, 8, 8, 0.33, 0.33);
  testMcCormickApproxForward(0, 1, 0, 1, 8, 8, 0.5, 0.5);
  testMcCormickApproxForward(0, 1, 0, 1, 8, 8, 0.5, 0.66);
  testMcCormickApproxForward(0, 1, 0, 1, 8, 8, 0.5, 0.5);


  printf("Bound [-1->1] tests:\n");
  printf("1x1 tests:\n");
  testMcCormickApproxForward(-1, 1, -1, 1, 1, 1, 0.33, 0.33);
  testMcCormickApproxForward(-1, 1, -1, 1, 1, 1, -0.5, 0.5);
  testMcCormickApproxForward(-1, 1, -1, 1, 1, 1, 0.5, -0.66);
  testMcCormickApproxForward(-1, 1, -1, 1, 1, 1, -0.5, -0.5);

  printf("2x2 tests:\n");
  testMcCormickApproxForward(-1, 1, -1, 1, 2, 2, 0.33, 0.33);
  testMcCormickApproxForward(-1, 1, -1, 1, 2, 2, -0.5, 0.5);
  testMcCormickApproxForward(-1, 1, -1, 1, 2, 2, 0.5, -0.66);
  testMcCormickApproxForward(-1, 1, -1, 1, 2, 2, -0.5, -0.5);

  printf("4x4 tests:\n");  
  testMcCormickApproxForward(-1, 1, -1, 1, 4, 4, 0.33, 0.33);
  testMcCormickApproxForward(-1, 1, -1, 1, 4, 4, -0.5, 0.5);
  testMcCormickApproxForward(-1, 1, -1, 1, 4, 4, 0.5, -0.66);
  testMcCormickApproxForward(-1, 1, -1, 1, 4, 4, -0.5, -0.5);

  printf("8x8 tests:\n");  
  testMcCormickApproxForward(-1, 1, -1, 1, 8, 8, 0.33, 0.33);
  testMcCormickApproxForward(-1, 1, -1, 1, 8, 8, -0.5, 0.5);
  testMcCormickApproxForward(-1, 1, -1, 1, 8, 8, 0.5, -0.66);
  testMcCormickApproxForward(-1, 1, -1, 1, 8, 8, -0.5, -0.5);


  return 0;
}
