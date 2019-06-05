#include "analysis.h"
#include "capturability.h"
#include "grid.h"
#include "loader.h"
#include "model.h"
#include "param.h"
#include "pendulum.h"
#include "polygon.h"
#include "vector.h"
#include <GL/freeglut.h>
#include <GL/glut.h>
#include <iostream>

using namespace std;
using namespace CA;

float x_min = -0.2;
float x_max = 0.2;
float y_min = -0.2;
float y_max = 0.2;

float x_resolution = 0.02;
float y_resolution = 0.02;

Vector2 point;

Model model("nao.xml");
Param param("analysis.xml");

Vector2 calcCop(State state_) {
  Grid grid(param);
  Capturability capturability(model, param);
  Vector2 point = state_.icp;
  Polygon polygon;
  polygon.setVertex(model.getVec("link", "foot_r"));

  Vector2 cop = polygon.getClosestPoint(point, polygon.getConvexHull());

  return cop;
}

void resize(int w, int h) {
  float s_x_min = -0.2, s_x_max = 0.2;
  float s_y_min = -0.2, s_y_max = 0.2;

  /* ウィンドウ全体をビューポートにする */
  glViewport(0, 0, w, h);

  /* 変換行列の初期化 */
  glLoadIdentity();

  /* スクリーン上の表示領域をビューポートの大きさに比例させる */
  glOrtho(-(s_x_max - s_x_min) * ((GLfloat)w) / 1000.0,
          (s_x_max - s_x_min) * ((GLfloat)w) / 1000.0,
          -(s_y_max - s_y_min) * ((GLfloat)h) / 1000.0,
          (s_y_max - s_y_min) * ((GLfloat)h) / 1000.0, -1.0, 1.0);
  glRotatef(90.0, 0.0, 0.0, 1.0);

  glFlush();
}

void grid() {
  glColor3f(0.0, 0.0, 0.0); //線の色
  for (int i = x_min / x_resolution; i <= x_max / x_resolution; i++) {
    if ((i % 10) == 0) {
      glLineWidth(2);
    } else {
      glLineWidth(1);
    }
    glBegin(GL_LINES);
    glVertex2f(x_resolution * i, y_min);
    glVertex2f(x_resolution * i, y_max);
    glEnd();
  }
  glColor3f(0.0, 0.0, 0.0); //線の色
  for (int i = y_min / y_resolution; i <= y_max / y_resolution; i++) {
    if ((i % 10) == 0) {
      glLineWidth(2);
    } else {
      glLineWidth(1);
    }
    glBegin(GL_LINES);
    glVertex2f(x_min, y_resolution * i);
    glVertex2f(x_max, y_resolution * i);
    glEnd();
  }

  std::vector<Vector2> v1 = model.getVec("link", "foot_r");
  glColor3f(0.5, 0.5, 0.5); //線の色
  glLineWidth(2);
  glBegin(GL_LINES);
  for (size_t i = 0; i < v1.size() - 1; i++) {
    glVertex2f(v1[i].x, v1[i].y);
    glVertex2f(v1[i + 1].x, v1[i + 1].y);
  }
  glEnd();

  Polygon polygon;
  polygon.setVertex(model.getVec("link", "foot_r"));
  std::vector<Vector2> v2 = polygon.getConvexHull();
  glColor3f(0.0, 0.0, 0.0); //線の色
  glLineWidth(2);
  glBegin(GL_LINES);
  for (size_t i = 0; i < v2.size() - 1; i++) {
    glVertex2f(v2[i].x, v2[i].y);
    glVertex2f(v2[i + 1].x, v2[i + 1].y);
  }
  glEnd();

  glPointSize(15);
  glColor3f(0.0, 0.0, 1.0);
  glBegin(GL_POINTS);
  glVertex2f(point.x, point.y);
  glEnd();

  glPointSize(10);
  glColor3f(1.0, 0.0, 0.0);
  glBegin(GL_POINTS);
  State state;
  state.icp.x = point.x;
  state.icp.y = point.y;
  glVertex2f(calcCop(state).x, calcCop(state).y);
  glEnd();
}

void display() {
  glClear(GL_COLOR_BUFFER_BIT);
  grid();
  glFlush();
}

void keyboard(unsigned char key, int x, int y) {
  switch (key) {
  case 'w':
    point.x += 0.01;
    display();
    break;
  case 'a':
    point.y += 0.01;
    display();
    break;
  case 'z':
    point.x -= 0.01;
    display();
    break;
  case 'd':
    point.y -= 0.01;
    display();
    break;
  case '\033': /* '\033' は ESC の ASCII コード */
    exit(0);
  default:
    break;
  }
}

int main(int argc, char *argv[]) {
  model.parse();
  param.parse();
  point.setPolar(0.1, -0.1);

  glutInit(&argc, argv);
  glutInitWindowPosition(100, 100);
  glutInitWindowSize(700, 700);
  glutInitDisplayMode(GLUT_RGBA);
  glutCreateWindow("CoP Assumption");
  glutDisplayFunc(display);
  glutReshapeFunc(resize);
  glutKeyboardFunc(keyboard);
  glClearColor(1.0, 1.0, 1.0, 1.0);
  glutMainLoop();
}
