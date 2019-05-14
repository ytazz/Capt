#ifndef __GL_WIDGET_H__
#define __GL_WIDGET_H__

#include <QOpenGLWidget>
#include <QWidget>
// #include <gl/GL.h>
// #include <gl/GLU.h>
#include "vector.h"
#include <GL/freeglut.h>
#include <GL/glut.h>

namespace CA {

struct GLPoint {
  Vector2 point;
  std::string color;
};

class GLWidget : public QOpenGLWidget {
public:
  GLWidget(QWidget *parent = nullptr, int width = 0, int height = 0);
  ~GLWidget();

protected:
  void initializeGL();
  void resizeGL(int w, int h);
  void paintGL();

  // for polar coordinate
  void paintPolarGrid();
  void paintPolar(float radius, float angle, const char *color_name);
  // for cartesian coordinate
  void paintCartesianGrid(float min, float max, float step);
  void paintCartesian(float x, float y, const char *color_name);

public slots:
  void setPolarGridRadius(float min, float max, float step,
                          const char *color_name);
  void setPolarGridAngle(float min, float max, float step,
                         const char *color_name);
  void setPolarPoint(Vector2 point, const char *color_name);
  void setPolarPoints(std::vector<Vector2> point, const char *color_name);
  void setPolarPolygon(std::vector<Vector2> vertex, const char *color_name);
  // paint function
  void paint();
  void reset();

public:
  // get RGB from color name
  void color(const char *color_name);

  // window size
  const float window_size;
  const float offset_angle;

  // for polar coordinate plot
  // grid
  bool paint_polar_r_grid, paint_polar_t_grid;
  const char *polar_grid_color;
  float polar_r_min, polar_r_max, polar_r_step;
  float polar_t_min, polar_t_max, polar_t_step;
  // point & polygon
  bool paint_polar_point;
  bool paint_polar_points;
  bool paint_polar_polygon;

  int windowWidth, windowHeight;
};

} // namespace CA

#endif // __GL_WIDGET_H__
