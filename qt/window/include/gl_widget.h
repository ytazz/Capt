#ifndef __GL_WIDGET_H__
#define __GL_WIDGET_H__

#include <QOpenGLWidget>
#include <QWidget>
// #include <gl/GL.h>
// #include <gl/GLU.h>
#include <GL/freeglut.h>
#include <GL/glut.h>

class GLWidget : public QOpenGLWidget {
public:
  GLWidget(QWidget *parent = nullptr, int width = 0, int height = 0);
  ~GLWidget();

protected:
  void initializeGL();
  void resizeGL(int w, int h);
  void paintGL();

public:
  // get RGB from color name
  void color(const char *color_name);

  // paint function
  void paint();

  // for polar coordinate
  bool paint_polar_grid;
  float polar_min, polar_max, polar_step;
  void setPolarGrid(float min, float max, float step);
  void paintPolarGrid(float min, float max, float step);
  void paintPolar(float radius, float angle, const char *color_name);
  // for cartesian coordinate
  void paintCartesianGrid(float min, float max, float step);
  void paintCartesian(float x, float y, const char *color_name);

  int windowWidth, windowHeight;
};

#endif // __GL_WIDGET_H__
