#ifndef __VECTOR_H__
#define __VECTOR_H__

#include <math.h>
#include <stdio.h>
#include <string>

namespace CA {

struct Vector2 {
  double x;
  double y;
  double norm();

  void print(std::string str);
  void print();

  void operator=(const Vector2 &v);

  Vector2 operator+(const Vector2 &v);
  Vector2 operator-(const Vector2 &v);
  double operator%(const Vector2 &v);
  Vector2 operator*(const double &d);
  double operator*(const Vector2 &v);
};

Vector2 operator*(const double &d, const Vector2 &v);

struct Vector3 {
  double x;
  double y;
  double z;
  double norm();
  double normXY();

  void print(std::string str);
  void print();

  void operator=(const Vector3 &v);

  Vector3 operator+(const Vector3 &v);

  Vector3 operator-(const Vector3 &v);

  Vector3 operator%(const Vector3 &v);
  Vector3 operator*(const double &d);
  double operator*(const Vector3 &v);
};

Vector3 operator*(const double &d, const Vector3 &v);

} // namespace CA

#endif // __VECTOR_H__
