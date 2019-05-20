#include "../include/vector.h"

namespace CA {

void Vector2::setPolar(float radius, float theta) {
  this->r = radius;
  this->t = theta;
  polarToCartesian();
}

void Vector2::setCartesian(float x, float y) {
  this->x = x;
  this->y = y;
  cartesianToPolar();
}

void Vector2::polarToCartesian() {
  this->x = this->r * cos(this->t);
  this->y = this->r * sin(this->t);
}

void Vector2::cartesianToPolar() {
  this->r = norm();
  this->t = atan2f(this->y, this->x);
  if (this->t < 0.0) {
    this->t += 2 * M_PI;
  }
}

float Vector2::norm() { return sqrt(x * x + y * y); }

void Vector2::printCartesian(std::string str) {
  printf("%s [ %lf, %lf ]\n", str.c_str(), this->x, this->y);
}

void Vector2::printPolar(std::string str) {
  printf("%s [ %lf, %lf ]\n", str.c_str(), this->r, this->t);
}

void Vector2::operator=(const Vector2 &v) {
  this->x = v.x;
  this->y = v.y;
  this->r = v.r;
  this->t = v.t;
}

Vector2 Vector2::operator+(const Vector2 &v) {
  Vector2 vec;
  vec.x = this->x + v.x;
  vec.y = this->y + v.y;
  vec.cartesianToPolar();
  return vec;
}

Vector2 Vector2::operator-(const Vector2 &v) {
  Vector2 vec;
  vec.x = this->x - v.x;
  vec.y = this->y - v.y;
  vec.cartesianToPolar();
  return vec;
}

float Vector2::operator%(const Vector2 &v) {
  float product;
  product = this->x * v.y - this->y * v.x;
  return product;
}

Vector2 Vector2::operator*(const float &d) {
  Vector2 vec;
  vec.x = this->x * d;
  vec.y = this->y * d;
  vec.r = this->r * d;
  vec.t = this->t;
  return vec;
}

float Vector2::operator*(const Vector2 &v) {
  float product = 0.0;
  product += this->x * v.x;
  product += this->y * v.y;
  return product;
}

Vector2 operator*(const float &d, const Vector2 &v) {
  Vector2 vec;
  vec.x = v.x * d;
  vec.y = v.y * d;
  vec.r = v.r * d;
  vec.t = v.t;
  return vec;
}

} // namespace CA
