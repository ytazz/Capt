#ifndef __VECTOR_H__
#define __VECTOR_H__

#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

namespace Capt {

struct Vector2 {

  Vector2(); // constructor
  ~Vector2();

  void clear();

  void setPolar(double radius, double theta);
  void setCartesian(double x, double y);

  void printPolar();
  void printPolar(std::string str);
  void printCartesian();
  void printCartesian(std::string str);

  double r, th;
  double x, y;
  double norm();

  Vector2 normal();

  void operator=(const Vector2 &v);

  Vector2 operator+(const Vector2 &v);
  Vector2 operator-(const Vector2 &v) const;
  double operator %(const Vector2 &v);
  Vector2 operator*(const double &d);
  double operator *(const Vector2 &v);
  Vector2 operator/(const double &d);

private:
  void cartesianToPolar();
  void polarToCartesian();

};

typedef Vector2 vec2_t;
typedef std::vector<Vector2> arr2_t;

Vector2 operator*(const double &d, const Vector2 &v);

} // namespace Capt

#endif // __VECTOR_H__
