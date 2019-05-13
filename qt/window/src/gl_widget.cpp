#include "gl_widget.h"

GLWidget::GLWidget(QWidget *parent, int width, int height)
    : QOpenGLWidget(parent) {
  // size
  windowWidth = width;
  windowHeight = height;
  setFixedSize(windowWidth, windowHeight);

  paint_polar_grid = false;
  polar_min = 0.0;
  polar_max = 0.0;
  polar_step = 0.0;
}

GLWidget::~GLWidget() {}

void GLWidget::initializeGL() {
  glClearColor(1.0, 1.0, 1.0, 1.0);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHTING);
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
  glEnable(GL_COLOR_MATERIAL);
}

void GLWidget::paintGL() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  if (paint_polar_grid)
    paintPolarGrid(polar_min, polar_max, polar_step);
}

void GLWidget::resizeGL(int w, int h) {
  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(45, (float)w / h, 0.01, 100.0);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0);
}

void GLWidget::color(const char *color_name) {
  if (strcmp(color_name, "red") == 0)
    glColor3f(1.0, 0.0, 0.0);
  if (strcmp(color_name, "green") == 0)
    glColor3f(0.0, 1.0, 0.0);
  if (strcmp(color_name, "blue") == 0)
    glColor3f(0.0, 0.0, 1.0);
  if (strcmp(color_name, "black") == 0)
    glColor3f(0.0, 0.0, 0.0);
  if (strcmp(color_name, "gray") == 0)
    glColor3f(0.5, 0.5, 0.5);
  if (strcmp(color_name, "yellow") == 0)
    glColor3f(1.0, 1.0, 0.0);
  if (strcmp(color_name, "white") == 0)
    glColor3f(0.0, 0.0, 0.0);
}

void GLWidget::paint() { paintGL(); }

void GLWidget::setPolarGrid(float min, float max, float step) {
  paint_polar_grid = true;
  polar_min = min;
  polar_max = max;
  polar_step = step;
}

void GLWidget::paintPolarGrid(float min, float max, float step) {
  const int resolution = 100;
  glBegin(GL_LINES);
  color("gray");
  for (int j = 0; j < 10; j++) {
    float radius = 0.2 * (j + 1);
    for (int i = 0; i < resolution; i++) {
      float theta = 2 * 3.1415926 * (float)i / (resolution);
      glVertex3f(radius * cos(theta), radius * sin(theta), 0);
      theta = 2 * 3.1415926 * (float)(i + 1) / (resolution);
      glVertex3f(radius * cos(theta), radius * sin(theta), 0);
    }
  }
  glEnd();
}
