#include "gl_widget.h"

namespace CA {

GLWidget::GLWidget(QWidget *parent, int width, int height)
    : QOpenGLWidget(parent), pi(M_PI), window_size(2.0), offset_angle(M_PI_2),
      circle_resolution(100) {
  // size
  window_width = width;
  window_height = height;
  setFixedSize(window_width, window_height);

  paint_polar_r_grid = false;
  polar_r_min = 0.0;
  polar_r_max = 0.0;
  polar_r_step = 0.0;
  paint_polar_t_grid = false;
  polar_t_min = 0.0;
  polar_t_max = 0.0;
  polar_t_step = 0.0;

  paint_point = false;
  paint_line = false;
  paint_circle = false;
  paint_arc = false;

  point.clear();
  line.clear();
  circle.clear();
  arc.clear();
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
    paintPoint();
  if (paint_line)
    paintLine();
  if (paint_circle)
    paintCircle();
  if (paint_arc)
    paintArc();
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

void GLWidget::color(const char *color) {
  if (strcmp(color, "red") == 0)
    glColor3f(1.0, 0.0, 0.0);
  if (strcmp(color, "green") == 0)
    glColor3f(0.0, 1.0, 0.0);
  if (strcmp(color, "blue") == 0)
    glColor3f(0.0, 0.0, 1.0);
  if (strcmp(color, "black") == 0)
    glColor3f(0.0, 0.0, 0.0);
  if (strcmp(color, "gray") == 0)
    glColor3f(0.5, 0.5, 0.5);
  if (strcmp(color, "yellow") == 0)
    glColor3f(1.0, 1.0, 0.0);
  if (strcmp(color, "white") == 0)
    glColor3f(0.0, 0.0, 0.0);
}

void GLWidget::paint() { update(); }

void GLWidget::reset() {
  paint_point = false;
  paint_line = false;
  paint_circle = false;
  paint_arc = false;
  point.clear();
  line.clear();
  circle.clear();
  arc.clear();

  update();
}

void GLWidget::setPolarGridRadius(float min, float max, float step,
                                  const char *color) {
  paint_polar_r_grid = true;

  polar_r_min = min;
  polar_r_max = max;
  polar_r_step = step;
  polar_grid_color = color;
}

void GLWidget::setPolarGridAngle(float min, float max, float step,
                                 const char *color) {
  paint_polar_t_grid = true;

  polar_t_min = min;
  polar_t_max = max;
  polar_t_step = step;
  polar_grid_color = color;
}

void GLWidget::setPoint(Vector2 point, const char *color) {
  std::vector<Vector2> vec;
  vec.push_back(point);
  setPoints(vec, color);
}

void GLWidget::setPoints(std::vector<Vector2> point, const char *color) {
  GLPoint gl_point;
  gl_point.color = color;
  for (size_t i = 0; i < point.size(); i++) {
    gl_point.point.push_back(point[i]);
  }
  this->point.push_back(gl_point);

  paint_point = true;
}

void GLWidget::setLine(std::vector<Vector2> vertex, const char *color) {
  GLPoint gl_point;
  gl_point.color = color;
  for (size_t i = 0; i < vertex.size(); i++) {
    gl_point.point.push_back(vertex[i]);
  }
  this->line.push_back(gl_point);

  paint_line = true;
}

void GLWidget::setCircle(Vector2 center, float radius, const char *color) {
  GLCircle gl_circle;
  gl_circle.center = center;
  gl_circle.radius = radius;
  gl_circle.color = color;

  circle.push_back(gl_circle);

  paint_circle = true;
}

void GLWidget::setArc(Vector2 center, float radius, float start_angle,
                      float end_angle, const char *color) {
  GLArc gl_arc;
  gl_arc.center = center;
  gl_arc.radius = radius;
  gl_arc.start_angle = start_angle;
  gl_arc.end_angle = end_angle;
  gl_arc.color = color;

  arc.push_back(gl_arc);

  paint_arc = true;
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
  glLineWidth(2);
  glBegin(GL_LINES);
  color(polar_grid_color);
  // radius-axis (line)
  bool continue_flag = true;
  float theta = polar_t_min;
  while (continue_flag) {
    glPolar2f(polar_r_min, theta + offset_angle);
    glPolar2f(polar_r_max, theta + offset_angle);
    theta += polar_t_step;
    if (theta > polar_t_max + polar_t_step / 2.0) {
      continue_flag = false;
    }
  }
  // angle-axis (circle)
  continue_flag = true;
  float radius = polar_r_min;
  while (continue_flag) {
    for (int i = 0; i < circle_resolution; i++) {
      theta = polar_t_min +
              (polar_t_max - polar_t_min) * (float)(i) / circle_resolution;
      glPolar2f(radius, theta + offset_angle);
      theta = polar_t_min +
              (polar_t_max - polar_t_min) * (float)(i + 1) / circle_resolution;
      glPolar2f(radius, theta + offset_angle);
    }
    radius += polar_r_step;
    if (radius > polar_r_max + polar_r_step / 2.0) {
      continue_flag = false;
    }
  }
  glEnd();
}

void GLWidget::paintPoint() {
  glPointSize(10);
  glEnable(GL_POINT_SMOOTH);
  glBegin(GL_POINTS);
  for (size_t i = 0; i < point.size(); i++) {
    color(point[i].color);
    for (size_t j = 0; j < point[i].point.size(); j++) {
      glCartesian2f(point[i].point[j].x, point[i].point[j].y);
    }
  }
  glEnd();
}

void GLWidget::paintLine() {
  glLineWidth(5);
  glBegin(GL_LINES);
  for (size_t i = 0; i < line.size(); i++) {
    color(line[i].color);
    for (size_t j = 0; j < line[i].point.size() - 1; j++) {
      glCartesian2f(line[i].point[j].x, line[i].point[j].y);
      glCartesian2f(line[i].point[j + 1].x, line[i].point[j + 1].y);
    }
  }
  glEnd();
}

void GLWidget::paintArc() {
  glLineWidth(2);
  glBegin(GL_LINES);
  for (size_t i = 0; i < arc.size(); i++) {
    color(arc[i].color);
    float theta = arc[i].start_angle;
    for (int j = 0; j < circle_resolution; j++) {
      glCartesian2f(arc[i].center.x + arc[i].radius * cos(theta),
                    arc[i].center.x + arc[i].radius * sin(theta));
      theta += (arc[i].end_angle - arc[i].start_angle) / circle_resolution;
      glCartesian2f(arc[i].center.x + arc[i].radius * cos(theta),
                    arc[i].center.x + arc[i].radius * sin(theta));
    }
  }

  paint_arc = true;
}

void GLWidget::paintCircle() {
  glLineWidth(2);
  glBegin(GL_LINES);
  for (size_t i = 0; i < circle.size(); i++) {
    color(circle[i].color);
    float theta = 0.0;
    for (int j = 0; j < circle_resolution; j++) {
      glCartesian2f(circle[i].center.x + circle[i].radius * cos(theta),
                    circle[i].center.x + circle[i].radius * sin(theta));
      theta += 2 * pi / circle_resolution;
      glCartesian2f(circle[i].center.x + circle[i].radius * cos(theta),
                    circle[i].center.x + circle[i].radius * sin(theta));
    }
  }

  paint_circle = true;
}

} // namespace CA
