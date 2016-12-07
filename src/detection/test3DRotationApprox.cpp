#include <stdexcept>
#include <iostream>

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
DecisionVariableMatrixX flatten_MxN( const DecisionVariableMatrixX& x ){
  DecisionVariableMatrixX ret(x.rows()*x.cols(), 1);
  for (int i=0; i<x.rows(); i++){ // for each row, paste that row, in order,
                                  // as elems in the new column vector
    ret.block(i*x.cols(), 0, x.cols(), 1) = x.block(i, 0, 1, x.cols()).transpose();
  }
  return ret;
}

void add_McCormick_envelope(MathematicalProgram& prog, 
                              DecisionVariableScalar& w,
                              DecisionVariableScalar& x, 
                              DecisionVariableScalar& y, 
                              string& corename,
                              double xL, 
                              double xH, 
                              double yL, 
                              double yH, 
                              int M_x,
                              int M_y){
  DecisionVariableMatrixX x_mat(1,1); x_mat(0,0) = x;
  DecisionVariableMatrixX y_mat(1,1); y_mat(0,0) = y;
  DecisionVariableMatrixX w_mat(1,1); w_mat(0,0) = w;

  // Add binary variables for the region we are in
  const double kStepSizeX =  (xH - xL) / (double)M_x;
  const double kStepSizeY =  (yH - yL) / (double)M_y;

  auto z_uv = prog.AddBinaryVariables(M_x, M_y, (corename + "_z").c_str());
  // and constrain that we can be in one at a time
  prog.AddLinearEqualityConstraint(MatrixXd::Ones(1, M_x*M_y), MatrixXd::Ones(1, 1), {flatten_MxN(z_uv)});

  // Create indicator variables xhat and yhat for each subsection
  auto x_hat = prog.AddContinuousVariables(M_x, M_y, (corename + string("_xhat")).c_str());
  auto y_hat = prog.AddContinuousVariables(M_x, M_y, (corename + string("_yhat")).c_str());
  // They must sum to their respective variable...
  prog.AddLinearEqualityConstraint(MatrixXd::Ones(1, 1+M_x*M_y), MatrixXd::Ones(1, 1), {x_mat, flatten_MxN(x_hat)});
  prog.AddLinearEqualityConstraint(MatrixXd::Ones(1, 1+M_x*M_y), MatrixXd::Ones(1, 1), {y_mat, flatten_MxN(y_hat)});
  // And respect the range of the region they represent -- which may force to zero if the region isn't active
  // Implemented as a bunch of upper and lower bounds
  MatrixXd A_region_bounds_xhat(M_x*M_y*2, M_x*M_y + M_x*M_y);
  MatrixXd A_region_bounds_yhat(M_x*M_y*2, M_x*M_y + M_x*M_y);
  MatrixXd lb_zero = MatrixXd::Zero(M_x*M_y*2, 1);
  MatrixXd ub_inf =  MatrixXd::Constant(M_x*M_y*2, 1, std::numeric_limits<double>::infinity());
  int k=0;
  for (int u=0; u<M_x; u++){
    for (int v=0; v<M_y; v++){
      double xL_uv = xL + u*kStepSizeX;
      double xH_uv = xL + (u+1)*kStepSizeX;
      double yL_uv = yL + v*kStepSizeY;
      double yH_uv = yL + (v+1)*kStepSizeY;
      // z(u,v) * xL(u,v) <= x_hat(u,v) <= z(u,v) * xH(u,v)
      A_region_bounds_xhat(2*k, k) = 1.0; // xhat - z(u,v) * xL(u,v) >= 0
      A_region_bounds_xhat(2*k, M_x*M_y+k) = -xL_uv;
      A_region_bounds_yhat(2*k, k) = 1.0; // yhat - z(u,v) * yL(u,v) >= 0
      A_region_bounds_yhat(2*k, M_x*M_y+k) = -yL_uv;

      A_region_bounds_xhat(2*k+1, k) = -1.0; // z(u,v) * xH(u,v) - xhat >= 0
      A_region_bounds_xhat(2*k+1, M_x*M_y+k) = xH_uv;
      A_region_bounds_yhat(2*k+1, k) = -1.0; // z(u,v) * yH(u,v) - yhat >= 0
      A_region_bounds_yhat(2*k+1, M_x*M_y+k) = yH_uv;
      k++;
    }
  }
  prog.AddLinearConstraint(A_region_bounds_xhat, lb_zero, ub_inf, {flatten_MxN(x_hat), flatten_MxN(z_uv)});
  prog.AddLinearConstraint(A_region_bounds_yhat, lb_zero, ub_inf, {flatten_MxN(y_hat), flatten_MxN(z_uv)});

  // And finally, constrain w by the four appropriate surfaces
  // Constraints w, xhats, yhats, z_uvs
  MatrixXd A_w_constraints(4, 1 + M_x*M_y + M_x*M_y + M_x*M_y);
  const int xhat_s = 1;
  const int yhat_s = xhat_s + M_x*M_y;
  const int zuv_s = yhat_s + M_x*M_y;
  lb_zero = MatrixXd::Zero(4, 1);
  ub_inf = MatrixXd::Constant(4, 1, std::numeric_limits<double>::infinity());
  k=0;
  for (int u=0; u<M_x; u++){
    for (int v=0; v<M_y; v++){
      double xL_uv = xL + u*kStepSizeX;
      double xH_uv = xL + (u+1)*kStepSizeX;
      double yL_uv = yL + v*kStepSizeY;
      double yH_uv = yL + (v+1)*kStepSizeY;
      // w >= sum_{uv} xL(u,v) * y_hat(u,v) + x_hat(u,v) * yL(u,v) - xL(u,v)*yL(u,v)*z(u,v)
      A_w_constraints(0, 0) = 1.0;
      A_w_constraints(0, xhat_s + k) = - yL_uv;
      A_w_constraints(0, yhat_s + k) = - xL_uv;
      //lb_zero(0) = -xL_uv * yL_uv;
      A_w_constraints(0, zuv_s  + k) = xL_uv * yL_uv;

      // w >= sum_{uv} xH(u,v) * y_hat(u,v) + x_hat(u,v) * yH(u,v) - xH(u,v)*yH(u,v)*z(u,v)
      A_w_constraints(1, 0) = 1.0;
      A_w_constraints(1, xhat_s + k) = - yH_uv;
      A_w_constraints(1, yhat_s + k) = - xH_uv;
      //lb_zero(1) = -xH_uv * yH_uv;
      A_w_constraints(1, zuv_s  + k) = xH_uv * yH_uv;

      // w <= sum_{uv} xH(u,v) * y_hat(u,v) + x_hat(u,v) * yL(u,v) - xH(u,v)*yL(u,v)*z(u,v)
      A_w_constraints(2, 0) = -1.0;
      A_w_constraints(2, xhat_s + k) = yL_uv;
      A_w_constraints(2, yhat_s + k) = xH_uv;
      //lb_zero(2) = xH_uv * yL_uv;
      A_w_constraints(2, zuv_s  + k) = -xH_uv * yL_uv;

      // w <= sum_{uv} xH(u,v) * y_hat(u,v) + x_hat(u,v) * yL(u,v) - xH(u,v)*yL(u,v)*z(u,v)
      A_w_constraints(3, 0) = -1.0;
      A_w_constraints(3, xhat_s + k) = yH_uv;
      A_w_constraints(3, yhat_s + k) = xL_uv;
      //lb_zero(3) = xL_uv * yH_uv;
      A_w_constraints(3, zuv_s  + k) = -xL_uv * yH_uv;

      k++;
    }
  }
  prog.AddLinearConstraint(A_w_constraints, lb_zero, ub_inf, {w_mat, flatten_MxN(x_hat), flatten_MxN(y_hat), flatten_MxN(z_uv)});
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

  // Solves earth movers problem for scene to model cloud
  // Crazy naive mixed integer question we're going to ask is
  // assignment of points in model to points in scene
  MathematicalProgram prog;

  auto R = prog.AddContinuousVariables(3, 3, "r");

  // Add core quaternion variables, ordered w x y z
  auto Q = prog.AddContinuousVariables(4, 1, "q");

  // Add variables for bilinear quaternion element products
  auto B = prog.AddContinuousVariables(10, 1, "b");

  // Constrain elements of rotation element by bilinear quaternion values
  // This constrains the 9 elements of the rotation matrix against the 
  // 10 bilinear terms in various combinations
  MatrixXd Aeq(9, 9 + 10);
  Aeq.setZero();
  MatrixXd beq(9, 1);
  beq.setZero();
  // build some utility inds to make writing this cleaner...
  int k=0;
  char qnames[5] = "wxyz";
  const int kNumRotVars = 9;
  const int kOffww = kNumRotVars + 0;
  const int kOffwx = kNumRotVars + 1;
  const int kOffwy = kNumRotVars + 2;
  const int kOffwz = kNumRotVars + 3;
  const int kOffxx = kNumRotVars + 4;
  const int kOffxy = kNumRotVars + 5;
  const int kOffxz = kNumRotVars + 6;
  const int kOffyy = kNumRotVars + 7;
  const int kOffyz = kNumRotVars + 8;
  const int kOffzz = kNumRotVars + 9;
  // Todo: I know you can do this formulaicaijasafcally...
  // R00 = 1 - 2 y^2 - 2 z^2 -> R00 + 2yy + 2zz = 1
  Aeq(k, 0) = 1.0;
  Aeq(k, kOffyy) = 2.0;
  Aeq(k, kOffzz) = 2.0;
  beq(k, 0) = 1.0;
  k++;
  // R01 = 2xy + 2wz -> R01 - 2xy - 2wz = 0
  Aeq(k, 1) = 1.0;
  Aeq(k, kOffxy) = -2.0;
  Aeq(k, kOffwz) = -2.0;
  beq(k, 0) = 0.0;
  k++;
  // R02 = 2xz - 2wy -> R02 - 2xz + 2wy = 0
  Aeq(k, 2) = 1.0;
  Aeq(k, kOffxz) = -2.0;
  Aeq(k, kOffwy) = 2.0;
  beq(k, 0) = 0.0;
  k++;
  // R10 = 2xy - 2wz -> R10 - 2xy + 2wz = 0
  Aeq(k, 3) = 1.0;
  Aeq(k, kOffxy) = -2;
  Aeq(k, kOffwz) = 2;
  beq(k, 0) = 0.0;
  k++;
  // R11 = 1 - 2xx - 2zz -> R11 + 2xx + 2zz = 1
  Aeq(k, 4) = 1.0;
  Aeq(k, kOffxx) = 2.0;
  Aeq(k, kOffzz) = 2.0;
  beq(k, 0) = 1.0;
  k++;
  // R12 = 2yz + 2wx -> r12 - 2yz - 2wx = 0
  Aeq(k, 5) = 1.0;
  Aeq(k, kOffyz) = -2.0;
  Aeq(k, kOffwx) = -2.0;
  beq(k, 0) = 0.0;
  k++;
  // R20 = 2xz + 2wy -> r20 - 2xz - 2wy = 0
  Aeq(k, 6) = 1.0;
  Aeq(k, kOffxz) = -2.0;
  Aeq(k, kOffwy) = -2.0;
  beq(k, 0) = 0.0;
  k++;
  // R21 = 2yz - 2wx -> r21 - 2yz + 2wx = 0
  Aeq(k, 7) = 1.0;
  Aeq(k, kOffyz) = -2.0;
  Aeq(k, kOffwx) = 2.0;
  beq(k, 0) = 0.0;
  k++;
  // R22 = 1 - 2xx - 2yy -> r22 + 2xx + 2yy = 1
  Aeq(k, 8) = 1.0;
  Aeq(k, kOffxx) = 2.0;
  Aeq(k, kOffyy) = 2.0;
  beq(k, 0) = 1.0;
  k++;
  prog.AddLinearEqualityConstraint(Aeq, beq, {flatten_MxN(R), B});


  // Now constrain xx + yy + zz + ww = 1
  prog.AddLinearEqualityConstraint(MatrixXd::Ones(1, 4), MatrixXd::Ones(1, 1), 
    {B.block<1,1>(0,0),B.block<1,1>(4,0),B.block<1,1>(7,0),B.block<1,1>(9,0)});


  // Finally, constrain each of the bilinear product pairs with their core quaternion variables
  k=0;
  for (int i=0; i<4; i++){
    for (int j=i; j<4; j++){
      // spawn new region selection variables
      string corename; corename += qnames[i]; corename += qnames[j];

      // select variable "x" and "y" out of quaternion
      auto x = Q(i, 0);
      auto y = Q(j, 0);
      // and select bilinear product "xy" variable
      auto xy = B(k,0);

      add_McCormick_envelope(prog, xy, x, y, corename,
                             -1.0, // xL
                             1.0,  // xH
                             -1.0, // yL
                             1.0,  // yH
                             4, 4); // M_x, M_y 
      k++;
    }
  }


  // Constrain quaternion value and see it propagate through to rotation
  MatrixXd desired_val = MatrixXd(4, 1);
  //desired_val << 0.258, 0.516, -0.258, -0.775;
  desired_val << 0.182, 0.365, 0.547, 0.730;
  cout << "Desired quaternion " << desired_val.transpose() << endl;
  printf("Target rot mat:\n \t(-0.666667 | 0.133333 | 0.733333\n\t0.666667 | -1/3 | 0.666667\n\t1/3 | 0.933333 | 0.133333)\n");
  prog.AddQuadraticErrorCost(MatrixXd::Identity(4, 4), desired_val, {Q});

  double now = getUnixTime();
  auto out = prog.Solve();
  string problem_string = "3drotapprox";
  double elapsed = getUnixTime() - now;
 
  printf("\tOptimized rotation:\n");
  printf("\t\t%f, %f, %f\n", R(0, 0).value(), R(0, 1).value(), R(0, 2).value());
  printf("\t\t%f, %f, %f\n", R(1, 0).value(), R(1, 1).value(), R(1, 2).value());
  printf("\t\t%f, %f, %f\n", R(2, 0).value(), R(2, 1).value(), R(2, 2).value());
  printf("\tOptimized quaternion:\n\t%f, %f, %f, %f\n", Q(0, 0).value(), Q(1, 0).value(), Q(2, 0).value(), Q(3, 0).value());
  printf("\tOptimized quaternion bilinear products:\n");
  k = 0;
  for (int i=0; i<4; i++){
    for (int j=i; j<4; j++){
      printf("\t\t%c%c:%f\n", qnames[i], qnames[j], B(k).value());
      k++;
    }
  }
  //prog.PrintSolution();

  printf("Problem %s solved with ret code %d with %lu variables in %f seconds\n", problem_string.c_str(), out, prog.num_vars(), elapsed);

  return 0;
}
