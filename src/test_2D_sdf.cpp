#include <stdexcept>
#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "sdf_2d_functions.hpp"

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
  printf("\n\n\n1D CASE:\n");
  VectorXd test_f(10);
  test_f << 0., INF, INF, INF, 0.0, 0.0, 0.0, INF, INF, INF; 
  cout << test_f.transpose() << endl;
  VectorXd d;
  VectorXi v;
  printf("Start df_1d1d with %ld rows\n", test_f.rows());
  df_1d(test_f, d, v);
  printf("Leave df_1d\n");
  cout << d.transpose() << endl;
  cout << v.transpose() << endl;
  cout << endl;

  printf("\n\n\n2D CASE:\n");

  MatrixXd test_f_2d(4, 4);
  test_f_2d << INF, INF, INF, INF,
            INF, INF, INF, INF,
            INF, INF, INF, INF,
            INF, INF, INF, INF,
  cout << test_f_2d << endl;
  MatrixXd d_2d;
  MatrixXi v_2d_row;
  MatrixXi v_2d_col;
  printf("Start df_2d with %ld rows, %ld cols\n", test_f_2d.rows(), test_f_2d.cols());
  df_2d(test_f_2d, d_2d, v_2d_row, v_2d_col);
  printf("Leave df_2d\n");
  cout << d_2d << endl;
  for (int i=0; i<test_f_2d.rows(); i++){
    for (int j=0; j<test_f_2d.cols(); j++){
      printf("(%d,%d)", v_2d_row(i,j), v_2d_col(i, j));
    }
    printf("\n");
  }

  printf("\n\n\nBIGGER 2D CASE:\n");
  MatrixXd test_f_2d_big = MatrixXd::Constant(15, 15, INF);
  test_f_2d_big(12, 12) = 0.0;
  test_f_2d_big(5, 12) = 0.0;
  test_f_2d_big(3, 1) = 0.0;
  test_f_2d_big(0, 0) = 0.0;
  cout << test_f_2d_big << endl;
  MatrixXd d_2d_big;
  MatrixXi v_2d_row_big;
  MatrixXi v_2d_col_big;
  printf("Start df_2d with %ld rows, %ld cols\n", test_f_2d_big.rows(), test_f_2d_big.cols());
  df_2d(test_f_2d_big, d_2d_big, v_2d_row_big, v_2d_col_big);
  printf("Leave df_2d\n");
  cout << d_2d_big << endl;
  for (int i=0; i<test_f_2d_big.rows(); i++){
    for (int j=0; j<test_f_2d_big.cols(); j++){
      printf("(%2d,%2d)", v_2d_row_big(i,j), v_2d_col_big(i, j));
    }
    printf("\n");
  }

  printf("\n\n\nHUGE 2D CASE:\n");
  MatrixXd test_f_2d_huge = MatrixXd::Constant(640, 480, INF);
  test_f_2d_huge(12, 12) = 0.0;
  test_f_2d_huge(5, 12) = 0.0;
  test_f_2d_huge(3, 1) = 0.0;
  test_f_2d_huge(0, 0) = 0.0;
  MatrixXd d_2d_huge;
  MatrixXi v_2d_row_huge;
  MatrixXi v_2d_col_huge;
  printf("Start df_2d with %ld rows, %ld cols\n", test_f_2d_huge.rows(), test_f_2d_huge.cols());
  df_2d(test_f_2d_huge, d_2d_huge, v_2d_row_huge, v_2d_col_huge);
  printf("Leave df_2d\n");
  cout << d_2d_huge.block<12, 12>(100, 100) << endl;

  return 0;
}