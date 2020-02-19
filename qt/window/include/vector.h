#ifndef __VECTOR_H__
#define __VECTOR_H__

#include <math.h>
#include <stdio.h>
#include <string>

namespace CA {

struct Vector2 {
  void init();

  void setPolar(float radius, float theta);
  void setCartesian(float x, float y);

  void printPolar(std::string str);
  void printCartesian(std::string str);

  float r, th;
  float x, y;
  float norm();

  void operator=(const Vector2 &v);

  Vector2 operator+(const Vector2 &v);
  Vector2 operator-(const Vector2 &v);
  float operator%(const Vector2 &v);
  Vector2 operator*(const float &d);
  float operator*(const Vector2 &v);

private:
  void cartesianToPolar();
  void polarToCartesian();
};

Vector2 operator*(const float &d, const Vector2 &v);

} // namespace CA

#endif // __VECTOR_H__
