#include "vector.cuh"

namespace GPGPU {

__device__ Vector2::Vector2() { clear(); }

__device__ Vector2::~Vector2() {}

__device__ void Vector2::clear() {
  this->r = 0.0;
  this->th = 0.0;
  this->x = 0.0;
  this->y = 0.0;
}

__device__ void Vector2::setPolar(float radius, float theta) {
  this->r = radius;
  this->th = theta;
  polarToCartesian();
}

__device__ void Vector2::setCartesian(float x, float y) {
  this->x = x;
  this->y = y;
  cartesianToPolar();
}

__device__ void Vector2::polarToCartesian() {
  this->x = this->r * cos(this->th);
  this->y = this->r * sin(this->th);
}

__device__ void Vector2::cartesianToPolar() {
  this->r = norm();
  this->th = atan2f(this->y, this->x);
  if (this->th < 0.0) {
    this->th += 2 * M_PI;
  }
}

__device__ float Vector2::norm() { return sqrt(x * x + y * y); }

__device__ Vector2 Vector2::normal() {
  Vector2 normal_vector;
  // rotate -90 deg around +z direction
  normal_vector.setCartesian(this->y, -this->x);
  return normal_vector;
}

__device__ void Vector2::operator=(const Vector2 &v) {
  this->x = v.x;
  this->y = v.y;
  this->r = v.r;
  this->th = v.th;
}

__device__ Vector2 Vector2::operator+(const Vector2 &v) {
  Vector2 vec;
  vec.x = this->x + v.x;
  vec.y = this->y + v.y;
  vec.cartesianToPolar();
  return vec;
}

__device__ Vector2 Vector2::operator-(const Vector2 &v) const {
  Vector2 vec;
  vec.x = this->x - v.x;
  vec.y = this->y - v.y;
  vec.cartesianToPolar();
  return vec;
}

__device__ float Vector2::operator%(const Vector2 &v) {
  float product;
  product = this->x * v.y - this->y * v.x;
  return product;
}

__device__ Vector2 Vector2::operator*(const float &d) {
  Vector2 vec;
  vec.x = this->x * d;
  vec.y = this->y * d;
  vec.r = this->r * d;
  vec.th = this->th;
  return vec;
}

__device__ float Vector2::operator*(const Vector2 &v) {
  float product = 0.0;
  product += this->x * v.x;
  product += this->y * v.y;
  return product;
}

__device__ Vector2 Vector2::operator/(const float &d) {
  Vector2 vec;
  vec.x = this->x / d;
  vec.y = this->y / d;
  vec.r = this->r / d;
  vec.th = this->th;
  return vec;
}

__device__ Vector2 operator*(const float &d, const Vector2 &v) {
  Vector2 vec;
  vec.x = v.x * d;
  vec.y = v.y * d;
  vec.r = v.r * d;
  vec.th = v.th;
  return vec;
}

} // namespace GPGPU
