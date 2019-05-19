#ifndef __GL_WIDGET_H__
#define __GL_WIDGET_H__

#include "vector.h"
#include <GL/freeglut.h>
#include <GL/glut.h>
#include <QOpenGLWidget>
#include <QWidget>
#include <math.h>

namespace CA {

struct GLPoint {
  std::vector<Vector2> point;
  const char *color;
};

struct GLCircle {
  Vector2 center;
  float radius;
  const char *color;
};

struct GLArc {
  Vector2 center;
  float radius;
  float start_angle;
  float end_angle;
  const char *color;
};

class GLWidget : public QOpenGLWidget {
public:
  GLWidget(QWidget *parent = nullptr, int width = 0, int height = 0);
  ~GLWidget();

protected:
  // OpenGL function
  void initializeGL();
  void resizeGL(int w, int h);
  void paintGL();

  // paint function
  // common
  void paintPoint();
  void paintLine();
  void paintCircle();
  void paintArc();
  // for polar coordinate
  void paintPolarGrid();
  // for cartesian coordinate
  void paintCartesianGrid();

public slots:
  // set point & line parameters
  // common
  void setPoint(Vector2 point, const char *color);
  void setPoints(std::vector<Vector2> point, const char *color);
  void setLine(std::vector<Vector2> vertex, const char *color);
  void setCircle(Vector2 center, float radius, const char *color);
  void setArc(Vector2 center, float radius, float start_angle, float end_angle,
              const char *color);
  // polar
  void setPolarGridRadius(float min, float max, float step, const char *color);
  void setPolarGridAngle(float min, float max, float step, const char *color);
  // cartesian
  void setCartesianGridX(float min, float max, float step, const char *color);
  void setCartesianGridY(float min, float max, float step, const char *color);

  // paint function
  void paint();
  // reset all parameters
  void reset();

public:
  // size of this window
  int window_width, window_height;

  // pi
  const float pi;

  // plot a vector
  void glPolar2f(const float radius, const float theta);
  void glCartesian2f(const float x, const float y);

  // get RGB from color name
  void color(const char *color);

  // window size
  const float window_size;

  // offset x-y angle
  const float offset_angle;

  // resolution of circle
  const int circle_resolution;

  // paint flag
  bool paint_point;
  bool paint_line;
  bool paint_circle;
  bool paint_arc;
  bool paint_polar_r_grid, paint_polar_t_grid;
  bool paint_cartesian_x_grid, paint_cartesian_y_grid;

  // for polar coordinate plot
  // grid
  const char *polar_grid_color;
  float polar_r_min, polar_r_max, polar_r_step;
  float polar_t_min, polar_t_max, polar_t_step;

  // memory
  std::vector<GLPoint> point;
  std::vector<GLPoint> line;
  std::vector<GLCircle> circle;
  std::vector<GLArc> arc;
};

} // namespace CA

#endif // __GL_WIDGET_H__
