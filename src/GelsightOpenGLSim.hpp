#ifndef GELSIGHT_OPENGL_SIM_H
#define GELSIGHT_OPENGL_SIM_H

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include <vector>
#include <Eigen/Dense>

#include "drake/multibody/shapes/drake_shapes.h"


class GelsightOpenGLSim {
public:
  GelsightOpenGLSim(const char * filename, double object_scale, int w, int h);
  ~GelsightOpenGLSim() {};

  Eigen::MatrixXd renderGelsight(Eigen::Vector3d xyz_sensor, Eigen::Vector3d xyz_lookat, Eigen::Vector3d xyz_up);

private:
  int w_, h_;

  int model_num_vertices_;
  int model_num_triangles_;
  GLuint model_vbo_verts_;
  GLuint model_vbo_triangles_;
  GLuint va_id_;
  float object_scale_;

  double gelsight_depth_ = 0.002;
  double gelsight_width_ = 0.025;
  double gelsight_height_ = 0.025;
};

#endif
