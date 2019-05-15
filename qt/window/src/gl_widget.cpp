#include "gl_widget.h"

namespace CA {

GLWidget::GLWidget(QWidget *parent, int width, int height)
    : QOpenGLWidget(parent), window_size(2.0), offset_angle(M_PI_2) {
  // size
  windowWidth = width;
  windowHeight = height;
  setFixedSize(windowWidth, windowHeight);

  paint_polar_r_grid = false;
  polar_r_min = 0.0;
  polar_r_max = 0.0;
  polar_r_step = 0.0;
  paint_polar_t_grid = false;
  polar_t_min = 0.0;
  polar_t_max = 0.0;
  polar_t_step = 0.0;

  paint_point = false;
  paint_polygon = false;

  point.clear();
  polygon.clear();
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
  if (paint_point)
    paintPolarPoint();
  if (paint_polygon)
    paintPolarPolygon();
  if (paint_polar_r_grid && paint_polar_t_grid)
    paintPolarGrid();
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

void GLWidget::paint() { update(); }

void GLWidget::reset() {
  paint_point = false;
  paint_polygon = false;
  update();
}

void GLWidget::setPolarGridRadius(float min, float max, float step,
                                  const char *color_name) {
  paint_polar_r_grid = true;

  polar_r_min = min;
  polar_r_max = max;
  polar_r_step = step;
  polar_grid_color = color_name;
}

void GLWidget::setPolarGridAngle(float min, float max, float step,
                                 const char *color_name) {
  paint_polar_t_grid = true;

  polar_t_min = min;
  polar_t_max = max;
  polar_t_step = step;
  polar_grid_color = color_name;
}

void GLWidget::setPoint(Vector2 point, const char *color_name) {
  std::vector<Vector2> vec;
  vec.push_back(point);
  setPoints(vec, color_name);
}
void GLWidget::setPoints(std::vector<Vector2> point, const char *color_name) {
  std::vector<GLPoint> p;
  GLPoint gp;
  for (size_t i = 0; i < point.size(); i++) {
    gp.color = color_name;
    gp.point = point[i];
    p.push_back(gp);
  }
  this->point.push_back(p);
  printf("%d\n", (int)point.size());

  paint_point = true;
}

void GLWidget::setPolygon(std::vector<Vector2> vertex, const char *color_name) {
  std::vector<GLPoint> p;
  GLPoint gp;
  for (size_t i = 0; i < vertex.size(); i++) {
    gp.color = color_name;
    gp.point = vertex[i];
    p.push_back(gp);
  }
  this->polygon.push_back(p);

  paint_polygon = true;
}

void GLWidget::glPolar2f(const float radius, const float theta) {
  float plot_x, plot_y;
  plot_x = radius * cos(theta + offset_angle);
  plot_y = radius * sin(theta + offset_angle);
  plot_x *= window_size / polar_r_max;
  plot_y *= window_size / polar_r_max;
  glVertex2f(plot_x, plot_y);
}

void GLWidget::glCartesian2f(const float x, const float y) {
  float plot_x, plot_y;
  plot_x = x * cos(offset_angle) - y * sin(offset_angle);
  plot_y = x * sin(offset_angle) + y * cos(offset_angle);
  plot_x *= window_size / polar_r_max;
  plot_y *= window_size / polar_r_max;
  glVertex2f(plot_x, plot_y);
}

void GLWidget::paintPolarGrid() {
  const int resolution = 100;
  glBegin(GL_LINES);
  color(polar_grid_color);
  // radius-axis (line)
  bool continue_flag = true;
  float theta = polar_t_min;
  while (continue_flag) {
    glVertex2f(
        polar_r_min * window_size / polar_r_max * cos(theta + offset_angle),
        polar_r_min * window_size / polar_r_max * sin(theta + offset_angle));
    glVertex2f(
        polar_r_max * window_size / polar_r_max * cos(theta + offset_angle),
        polar_r_max * window_size / polar_r_max * sin(theta + offset_angle));
    theta += polar_t_step;
    if (theta > polar_t_max + polar_t_step / 2.0) {
      continue_flag = false;
    }
  }
  // angle-axis (circle)
  continue_flag = true;
  float radius = polar_r_min;
  while (continue_flag) {
    for (int i = 0; i < resolution; i++) {
      theta =
          polar_t_min + (polar_t_max - polar_t_min) * (float)(i) / resolution;
      glVertex2f(radius * window_size / polar_r_max * cos(theta + offset_angle),
                 radius * window_size / polar_r_max *
                     sin(theta + offset_angle));
      theta = polar_t_min +
              (polar_t_max - polar_t_min) * (float)(i + 1) / resolution;
      glVertex2f(radius * window_size / polar_r_max * cos(theta + offset_angle),
                 radius * window_size / polar_r_max *
                     sin(theta + offset_angle));
    }
    radius += polar_r_step;
    if (radius > polar_r_max + polar_r_step / 2.0) {
      continue_flag = false;
    }
  }
  glEnd();
}

void GLWidget::paintPolarPoint() {
  glPointSize(10);
  glEnable(GL_POINT_SMOOTH);
  glBegin(GL_POINTS);
  for (size_t i = 0; i < point.size(); i++) {
    for (size_t j = 0; j < point[i].size(); j++) {
      color(point[i][j].color);
      glPolar2f(point[i][j].point.r, point[i][j].point.t);
    }
  }
  glEnd();
}

void GLWidget::paintPolarPolygon() {
  glBegin(GL_LINES);
  for (size_t i = 0; i < polygon.size(); i++) {
    for (size_t j = 0; j < polygon[i].size() - 1; j++) {
      color(polygon[i][j].color);
      glPolar2f(polygon[i][j].point.r, polygon[i][j].point.t);
      glPolar2f(polygon[i][j + 1].point.r, polygon[i][j + 1].point.t);
    }
  }
  glEnd();
}

} // namespace CA
