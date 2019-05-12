#include "../include/vector.h"

namespace CA {

double Vector2::norm() { return sqrt(x * x + y * y); }

void Vector2::print(std::string s) {
  printf("%s [ %lf, %lf ]\n", s.c_str(), x, y);
}

void Vector2::print() { print(""); }

void Vector2::operator=(const Vector2 &v) {
  this->x = v.x;
  this->y = v.y;
}

Vector2 Vector2::operator+(const Vector2 &v) {
  Vector2 vec;
  vec.x = this->x + v.x;
  vec.y = this->y + v.y;
  return vec;
}

Vector2 Vector2::operator-(const Vector2 &v) {
  Vector2 vec;
  vec.x = this->x - v.x;
  vec.y = this->y - v.y;
  return vec;
}

double Vector2::operator%(const Vector2 &v) {
  double product;
  product = this->x * v.y - this->y * v.x;
  return product;
}

Vector2 Vector2::operator*(const double &d) {
  Vector2 vec;
  vec.x = this->x * d;
  vec.y = this->y * d;
  return vec;
}

double Vector2::operator*(const Vector2 &v) {
  double product = 0.0;
  product += this->x * v.x;
  product += this->y * v.y;
  return product;
}

Vector2 operator*(const double &d, const Vector2 &v) {
  Vector2 vec;
  vec.x = v.x * d;
  vec.y = v.y * d;
  return vec;
}

double Vector3::norm() { return sqrt(x * x + y * y + z * z); }

double Vector3::normXY() { return sqrt(x * x + y * y); }

void Vector3::print(std::string s) {
  printf("%s [ %lf, %lf, %lf ]\n", s.c_str(), x, y, z);
}

void Vector3::print() { print(""); }

void Vector3::operator=(const Vector3 &v) {
  this->x = v.x;
  this->y = v.y;
  this->z = v.z;
}

Vector3 Vector3::operator+(const Vector3 &v) {
  Vector3 vec;
  vec.x = this->x + v.x;
  vec.y = this->y + v.y;
  vec.z = this->z + v.z;
  return vec;
}

Vector3 Vector3::operator-(const Vector3 &v) {
  Vector3 vec;
  vec.x = this->x - v.x;
  vec.y = this->y - v.y;
  vec.z = this->z - v.z;
  return vec;
}

Vector3 Vector3::operator%(const Vector3 &v) {
  Vector3 vec;
  vec.x = this->y * v.z - this->z * v.y;
  vec.y = this->z * v.x - this->x * v.z;
  vec.z = this->x * v.y - this->y * v.x;
  return vec;
}

Vector3 Vector3::operator*(const double &d) {
  Vector3 vec;
  vec.x = this->x * d;
  vec.y = this->y * d;
  vec.z = this->z * d;
  return vec;
}

double Vector3::operator*(const Vector3 &v) {
  double product = 0.0;
  product += this->x * v.x;
  product += this->y * v.y;
  product += this->z * v.z;
  return product;
}

Vector3 operator*(const double &d, const Vector3 &v) {
  Vector3 vec;
  vec.x = v.x * d;
  vec.y = v.y * d;
  vec.z = v.z * d;
  return vec;
}

} // namespace CA
