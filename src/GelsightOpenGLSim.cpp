#undef NDEBUG
#include "GelsightOpenGLSim.hpp"

#include <assert.h>
#include <cmath>
#include <math.h>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace Eigen;


GelsightOpenGLSim::GelsightOpenGLSim(const char * filename, double object_scale, int w, int h) :
    object_scale_(object_scale),
    w_(w),
    h_(h)
{
  printf("Trying to load file %s ... \n", filename);

  char * myargv[1];
  int myargc=1;
  myargv[0] = strdup("asdf");
  glutInit(&myargc, myargv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
  glutInitWindowSize(w_,h_);
  glutInitWindowPosition(0,0);
    /* create and set up a window */
  glutCreateWindow("GelsightOpenGLSim");

  glewInit();  

  glEnable( GL_DEPTH_TEST );

  DrakeShapes::Mesh mesh("gelsight sim mesh", filename);

  DrakeShapes::PointsVector vertices;
  DrakeShapes::TrianglesVector triangles;
  mesh.LoadObjFile(&vertices, &triangles);
  printf("\t Successfully loaded %lu verts and %lu triangles\n", vertices.size(), triangles.size());

  model_num_vertices_ = vertices.size();
  model_num_triangles_ = triangles.size();

  // For my sanity, I'm going to rip those eigen standard vector things
  // into tightly packed arrays that I know the internal alignment of.
  // I'm sure you could skip this and construct a VBO directly from the
  // Drake typedef'd things, but I don't want to figure that out today. 

  float * tight_vert_array = (float *) malloc( sizeof(float) * 3 * model_num_vertices_ );
  unsigned int * tight_triangle_array = (unsigned int *) malloc( sizeof(unsigned int) * 3 * model_num_triangles_);

  for (int i=0; i<model_num_vertices_; i++){
    tight_vert_array[i*3] = vertices[i][0]*object_scale_;
    tight_vert_array[i*3+1] = vertices[i][1]*object_scale_;
    tight_vert_array[i*3+2] = vertices[i][2]*object_scale_;
  }
  for (int i=0; i<model_num_triangles_; i++){
    tight_triangle_array[i*3] = (unsigned int) triangles[i][0];
    tight_triangle_array[i*3+1] = (unsigned int) triangles[i][1];
    tight_triangle_array[i*3+2] = (unsigned int) triangles[i][2];
  }

  // Vertex array object
  glGenVertexArrays(1, &va_id_);
  glBindVertexArray(va_id_);

  // Vertex buffer
  glGenBuffers(1, &model_vbo_verts_);
  glBindBuffer(GL_ARRAY_BUFFER, model_vbo_verts_);
  glBufferData(GL_ARRAY_BUFFER, model_num_vertices_ * 3 * sizeof(float), &tight_vert_array[0], GL_STATIC_DRAW);

  // Index buffer for the triangles
  glGenBuffers(1, &model_vbo_triangles_);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, model_vbo_triangles_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, model_num_triangles_ * 3 * sizeof(unsigned int), &tight_triangle_array[0], GL_STATIC_DRAW);

  free(tight_vert_array);
  free(tight_triangle_array); 
}


Eigen::MatrixXd GelsightOpenGLSim::renderGelsight(Eigen::Vector3d xyz_sensor, Eigen::Vector3d xyz_lookat, Eigen::Vector3d xyz_up){
  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

  const float distance_extra = 0.5; // putting near plane at 0 distance from projection center
                                    // causes numerical terribleness

  float zNear = distance_extra;
  float zFar = distance_extra+gelsight_depth_;

    /* define the projection transformation */
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(-gelsight_width_/2.0,gelsight_width_/2.0,
          -gelsight_height_/2.0,gelsight_height_/2.0,
          zNear,zFar);

  glMatrixMode( GL_MODELVIEW );
  glLoadIdentity();

  gluLookAt(xyz_sensor[0], xyz_sensor[1], xyz_sensor[2],
           xyz_lookat[0], xyz_lookat[1], xyz_lookat[2],
           xyz_up[0], xyz_up[1], xyz_up[2]);
  glTranslatef( 0.0, 0.0, distance_extra);

  // Draw the object
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, model_vbo_verts_);
  glVertexAttribPointer(
    0, // attribute 0
    3, // size
    GL_FLOAT, // type
    GL_FALSE, // normalized?
    0, // stride
    (void*)0  // offset
  );
  // Index buffer
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, model_vbo_triangles_);
  glDrawElements(
      GL_TRIANGLES,      // mode
      model_num_triangles_*3,    // count
      GL_UNSIGNED_INT,   // type
      (void*)0           // element array buffer offset
  );
  glDisableVertexAttribArray(0);

  // get the depth buffer out
  std::vector< GLfloat > depth( w_ * h_, 0 );
  glReadPixels( 0, 0, w_, h_, GL_DEPTH_COMPONENT, GL_FLOAT, &depth[0] ); 

  MatrixXd depthMat(w_, h_); // need tranpose at end of day
  // linearize depth
  // http://www.geeks3d.com/20091216/geexlab-how-to-visualize-the-depth-buffer-in-glsl/
  for( size_t i = 0; i < depth.size(); ++i )
  {
    //depthMat(i % h_, i / h_) = (double)(depth[i]);
    depthMat(i % w_, i / w_) = depth[i];
    //( 2.0 * zNear ) / ( zFar + zNear - depth[i] * ( zFar - zNear ) );
  }

  glutSwapBuffers();

  return depthMat;
}