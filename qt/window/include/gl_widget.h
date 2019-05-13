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

  int windowWidth, windowHeight;
};

#endif // __GL_WIDGET_H__