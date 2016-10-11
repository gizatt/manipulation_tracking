#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <vector>

#include "drake/systems/plants/shapes/DrakeShapes.h"

int num_vertices, num_triangles;
GLuint vbo_model_verts;
GLuint vbo_model_normals;
GLuint va_id;

// camera attributes
float viewerPosition[3]   = { 0.0, 0.0, -1.0 };
float viewerDirection[3]  = { 0.0, 0.0, 0.0 };
float viewerUp[3]     = { 0.0, 1.0, 0.0 };

// rotation values for the navigation
float navigationRotation[3] = { 0.0, 0.0, 0.0 };

// position of the mouse when pressed
int mousePressedX = 0, mousePressedY = 0;
float lastXOffset = 0.0, lastYOffset = 0.0, lastZOffset = 0.0;
// mouse button states
int leftMouseButtonActive = 0, middleMouseButtonActive = 0, rightMouseButtonActive = 0;
// modifier state
int shiftActive = 0, altActive = 0, ctrlActive = 0;

float zNear = 1.0;
float zFar = 1.5;

bool show_depth = false;

// Borrowed from Drake
void load_obj_to_FBO(const char* filename, float scale=1.0)
{

  printf("Trying to load file %s ... \n", filename);

  DrakeShapes::Mesh mesh("gelsight sim mesh", filename);

  DrakeShapes::PointsVector vertices;
  DrakeShapes::TrianglesVector triangles;
  mesh.LoadObjFile(&vertices, &triangles);
  printf("\t Successfully loaded %lu verts and %lu triangles\n", vertices.size(), triangles.size());

  num_vertices = vertices.size();
  num_triangles = triangles.size();

  // For my sanity, I'm going to rip those eigen standard vector things
  // into tightly packed arrays that I know the internal alignment of.
  // I'm sure you could skip this and construct a VBO directly from the
  // Drake typedef'd things, but I don't want to figure that out today. 

  float * tight_vert_array = (float *) malloc( sizeof(float) * 3 * num_vertices );
  unsigned int * tight_triangle_array = (unsigned int *) malloc( sizeof(unsigned int) * 3 * num_triangles);

  for (int i=0; i<num_vertices; i++){
    tight_vert_array[i*3] = vertices[i][0]*scale;
    tight_vert_array[i*3+1] = vertices[i][1]*scale;
    tight_vert_array[i*3+2] = vertices[i][2]*scale;
  }
  for (int i=0; i<num_triangles; i++){
    tight_triangle_array[i*3] = (unsigned int) triangles[i][0];
    tight_triangle_array[i*3+1] = (unsigned int) triangles[i][1];
    tight_triangle_array[i*3+2] = (unsigned int) triangles[i][2];
  }

  // Vertex array object
  glGenVertexArrays(1, &va_id);
  glBindVertexArray(va_id);

  // Vertex buffer
  glGenBuffers(1, &vbo_model_verts);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_model_verts);
  glBufferData(GL_ARRAY_BUFFER, num_vertices * 3 * sizeof(float), &tight_vert_array[0], GL_STATIC_DRAW);

  // Index buffer for the triangles
  glGenBuffers(1, &vbo_model_normals);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_model_normals);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, num_triangles * 3 * sizeof(unsigned int), &tight_triangle_array[0], GL_STATIC_DRAW);

  free(tight_vert_array);
  free(tight_triangle_array); 
}


void display () {

    int w = glutGet( GLUT_WINDOW_WIDTH );
    int h = glutGet( GLUT_WINDOW_HEIGHT );

  glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    /* define the projection transformation */
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  double ar = w / static_cast< double >( h );
  glOrtho(-1.0,1.0,-1.0,1.0,zNear,zFar);

  glMatrixMode( GL_MODELVIEW );
  glLoadIdentity();

  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);

  GLfloat lightpos[4] = { 5.0, 15.0, 10.0, 1.0 }; 
  glLightfv(GL_LIGHT0, GL_POSITION, lightpos);

  glTranslatef( viewerPosition[0], viewerPosition[1], viewerPosition[2] );
  // add navigation rotation

  glRotatef( navigationRotation[0], 1.0f, 0.0f, 0.0f );
  glRotatef( navigationRotation[1], 0.0f, 1.0f, 0.0f );

    /* draw scene */
  //glColor3ub( 255, 0, 0 );

  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_model_verts);
  glVertexAttribPointer(
    0, // attribute 0
    3, // size
    GL_FLOAT, // type
    GL_FALSE, // normalized?
    0, // stride
    (void*)0  // offset
  );

  // Index buffer
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_model_normals);
  glDrawElements(
      GL_TRIANGLES,      // mode
      num_triangles*3,    // count
      GL_UNSIGNED_INT,   // type
      (void*)0           // element array buffer offset
  );

  glDisableVertexAttribArray(0);

  if (show_depth){

    // get the depth buffer out
    std::vector< GLfloat > depth( w * h, 0 );
    glReadPixels( 0, 0, w, h, GL_DEPTH_COMPONENT, GL_FLOAT, &depth[0] ); 

    glDisable(GL_LIGHTING);
    glDisable(GL_LIGHT0);

      // linearize depth
      // http://www.geeks3d.com/20091216/geexlab-how-to-visualize-the-depth-buffer-in-glsl/
    // also find min / max that's less than 1
    float min = 1.0;
    float max = 0.0;
    float mean = 0.0;
    float stddev = 0.0;
    for( size_t i = 0; i < depth.size(); ++i )
    {
      depth[i] = ( 2.0 * zNear ) / ( zFar + zNear - depth[i] * ( zFar - zNear ) );
    }


    // apply that depth buffer to a texture, and draw than texture onto
    // a quad in front of the camera
    static GLuint tex = 0;
    if( tex > 0 )
      glDeleteTextures( 1, &tex );
    glGenTextures(1, &tex);
    glBindTexture( GL_TEXTURE_2D, tex);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexImage2D( GL_TEXTURE_2D, 0, GL_LUMINANCE, w, h, 0, GL_LUMINANCE, GL_FLOAT, &depth[0] );

    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();
    glOrtho( 0, w, 0, h, -1, 1 );

    glMatrixMode( GL_MODELVIEW );
    glLoadIdentity();

    glEnable( GL_TEXTURE_2D );
    glColor3ub( 255, 255, 255 );
    glScalef( 0.3, 0.3, 1 );
    glBegin( GL_QUADS );
    glTexCoord2i( 0, 0 );
    glVertex2i( 0, 0 );
    glTexCoord2i( 1, 0 );
    glVertex2i( w, 0 );
    glTexCoord2i( 1, 1 );
    glVertex2i( w, h);
    glTexCoord2i( 0, 1 );
    glVertex2i( 0, h );
    glEnd();
    glDisable( GL_TEXTURE_2D );
  }

  glutSwapBuffers();

}

void reshape ( int width, int height ) {

    /* define the viewport transformation */
  glViewport(0,0,width,height);
}


// mouse callback
void mouseFunc(int button, int state, int x, int y) {
  
  // get the modifiers
  switch (glutGetModifiers()) {

    case GLUT_ACTIVE_SHIFT:
      shiftActive = 1;
      break;
    case GLUT_ACTIVE_ALT:
      altActive = 1;
      break;
    case GLUT_ACTIVE_CTRL:
      ctrlActive  = 1;
      break;
    default:
      shiftActive = 0;
      altActive = 0;
      ctrlActive  = 0;
      break;
  }

  // get the mouse buttons
  if (button == GLUT_LEFT_BUTTON)
    if (state == GLUT_DOWN) {
      leftMouseButtonActive += 1;
    } else {
      leftMouseButtonActive -= 1;
    }

  else if (button == GLUT_MIDDLE_BUTTON)
    if (state == GLUT_DOWN) {
      middleMouseButtonActive += 1;
      lastXOffset = 0.0;
      lastYOffset = 0.0;
    } else
      middleMouseButtonActive -= 1;
  else if (button == GLUT_RIGHT_BUTTON)
    if (state == GLUT_DOWN) {
      rightMouseButtonActive += 1;
      lastZOffset = 0.0;
    } else
      rightMouseButtonActive -= 1;

//  if (altActive) {
    mousePressedX = x;
    mousePressedY = y;
//  }
}

//-----------------------------------------------------------------------------

void mouseMotionFunc(int x, int y) {
  
  float xOffset = 0.0, yOffset = 0.0, zOffset = 0.0;

  // navigation
//  if (altActive) {
  
    // rotatation
    if (leftMouseButtonActive) {

      navigationRotation[0] += ((mousePressedY - y) * 180.0f) / 200.0f;
      navigationRotation[1] += ((mousePressedX - x) * 180.0f) / 200.0f;

      mousePressedY = y;
      mousePressedX = x;

    }
    // panning
    else if (middleMouseButtonActive) {

      xOffset = (mousePressedX + x);
      if (!lastXOffset == 0.0) {
        viewerPosition[0] -= (xOffset - lastXOffset) / 100.0;
        viewerDirection[0]  -= (xOffset - lastXOffset) / 100.0;
      }
      lastXOffset = xOffset;

      yOffset = (mousePressedY + y);
      if (!lastYOffset == 0.0) {
        viewerPosition[1] += (yOffset - lastYOffset) / 100.0;
        viewerDirection[1]  += (yOffset - lastYOffset) / 100.0; 
      } 
      lastYOffset = yOffset;

    }
    // depth movement
    else if (rightMouseButtonActive) {
      zOffset = (mousePressedX + x);
      if (!lastZOffset == 0.0) {
        viewerPosition[2] -= (zOffset - lastZOffset) / 100.0;
        viewerDirection[2] -= (zOffset - lastZOffset) / 100.0;
        printf("Viewerpos: %f. ViewerDir: %f\n", viewerPosition[2], viewerDirection[2]);
      }
      lastZOffset = zOffset;
    }
//  }
    glutPostRedisplay();
}

void keyboardFunc(unsigned char key, int x, int y) {
  switch(key) {
    case 'd':
      show_depth = !show_depth;
      printf("Toggled depth to %d\n", show_depth);
      break;
    default:
      printf("key %c pressed\n", key);
  }
  glutPostRedisplay();
}



int main ( int argc, char * argv[] ) {

    /* initialize GLUT, using any commandline parameters passed to the 
       program */
  glutInit(&argc,argv);

    /* setup the size, position, and display mode for new windows */
  glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
  glutInitWindowSize(1024,768);
  glutInitWindowPosition(0,0);

    /* create and set up a window */
  glutCreateWindow("hello, teapot!");

  glewInit();

  glutDisplayFunc(display);
  glutReshapeFunc(reshape);

  glutKeyboardFunc  (keyboardFunc);
  glutMouseFunc   (mouseFunc);
  glutMotionFunc    (mouseMotionFunc);


  glEnable( GL_DEPTH_TEST );
//    glDepthFunc( GL_LESS );

  load_obj_to_FBO("/home/gizatt/drc/software/perception/manipulation_tracking/urdf/screwdriver_torx/625330_TX10_Garant.obj", 0.01);

    /* tell GLUT to wait for events */
  glutMainLoop();
}