#ifndef __sdf_2d_functions_h__
#define __sdf_2d_functions_h__

#define INF 1E16
static inline float square(float x) { return x * x; }

// Significant reference to code available from http://cs.brown.edu/~pff/dt/
// f: Nx1 binary image.
// d: will return populated with the distance from each pixel of the input image,
//     to the nearest 0-pixel of the input image
// v: will return populated with the index of the closest 0-pixel from the input image
//     for every pixel in the input image
void df_1d(const Eigen::VectorXd& f, Eigen::VectorXd& d, Eigen::VectorXi& mapping) {
  int n = f.rows();
  int *v = new int[n];
  d.resize(n);
  mapping.resize(n);
  float *z = new float[n+1];
  int k = 0;
  v[0] = 0;
  z[0] = -INF;
  z[1] = +INF;
  for (int q = 1; q <= n-1; q++) {
    float s  = ((f[q]+square(q))-(f[v[k]]+square(v[k])))/(2*q-2*v[k]);
    while (s <= z[k]) {
      k--;
      s  = ((f[q]+square(q))-(f[v[k]]+square(v[k])))/(2*q-2*v[k]);
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k+1] = +INF;
  }

  k = 0;
  for (int q = 0; q <= n-1; q++) {
    while (z[k+1] < q)
      k++;
    
    d[q] = square(q-v[k]) + f[v[k]];
    mapping(q) = v[k];

  }

  delete [] z;
  delete [] v;
}

// f: NxM binary image.
// d: will return populated with the distance from each pixel of the input image,
//     to the nearest 0-pixel of the input image
// v: will return populated with the index of the closest 0-pixel from the input image
//     for every pixel in the input image
void df_2d(const Eigen::MatrixXd& f, Eigen::MatrixXd& d, Eigen::MatrixXi& mapping_row, Eigen::MatrixXi& mapping_col) {
  int width = f.cols();
  int height = f.rows();

  d.resize(height, width);
  d = f;
  Eigen::MatrixXi mapping_row_tmp(height, width);
  mapping_row.resize(height, width);
  mapping_col.resize(height, width);
  for (int i=0; i<height; i++){
    for (int j=0; j<width; j++){
      mapping_row_tmp(i, j) = i;
      mapping_col(i, j) = j;
    }
  }

  // transform along columns
  for (int x = 0; x < width; x++) {
    Eigen::VectorXd d_thiscol = d.block(0, x, height, 1);
    Eigen::VectorXi mapping_row_thiscol = mapping_row_tmp.block(0, x, height, 1);
    df_1d(d.block(0, x, height, 1), d_thiscol, mapping_row_thiscol);
    d.block(0, x, height, 1) = d_thiscol;
    for (int y=0; y < height; y++){
      if (d(y, x) < INF)
        mapping_row_tmp(y, x) = mapping_row_thiscol(y);
    }
  }

  // transform along rows
  for (int y = 0; y < height; y++) {
    Eigen::VectorXd d_thisrow = d.block(y, 0, 1, width).transpose();
    Eigen::VectorXi mapping_col_thisrow = mapping_col.block(y, 0, 1, width).transpose();
    df_1d(d.block(y, 0, 1, width).transpose(), d_thisrow, mapping_col_thisrow);
    d.block(y, 0, 1, width) = d_thisrow.transpose();
    
    for (int x=0; x < width; x++){
      if (d(y, x) < INF){
        // copy over the correct column
        mapping_col(y, x) = mapping_col_thisrow(x);
      }
    }
  }

  // resolve all references
  for (int x=0; x < width; x++){
    for (int y=0; y < height; y++){
      mapping_row(y, x) = mapping_row_tmp(y, mapping_col(y, x));
    }
  }
}
#endif