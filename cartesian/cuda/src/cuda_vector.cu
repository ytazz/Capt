#include "cuda_vector.cuh"

namespace Cuda {

__device__ void Vector2::clear() {
  this->x = 0.0;
  this->y = 0.0;
}

__device__ void Vector2::set(double x, double y) {
  this->x = x;
  this->y = y;
}

__device__ double Vector2::norm() {
  return sqrt(x * x + y * y);
}

__device__ Vector2 Vector2::normal() {
  Vector2 normal_vector;
  // rotate -90 deg around +z direction
  normal_vector.set(this->y, -this->x);
  return normal_vector;
}

__device__ Vector2 &Vector2::operator=(const Vector2 &v) {
  this->x = v.x;
  this->y = v.y;
  return *this;
}

__device__ Vector2 Vector2::operator+(const Vector2 &v) {
  Vector2 vec;
  vec.set(this->x + v.x, this->y + v.y);
  return vec;
}

__device__ Vector2 Vector2::operator-(const Vector2 &v) {
  Vector2 vec;
  vec.set(this->x - v.x, this->y - v.y);
  return vec;
}

__device__ double Vector2::operator%(const Vector2 &v) {
  double product;
  product = this->x * v.y - this->y * v.x;
  return product;
}

__device__ Vector2 Vector2::operator*(const double &d) {
  Vector2 vec;
  vec.set(this->x * d, this->y * d);
  return vec;
}

__device__ double Vector2::operator*(const Vector2 &v) {
  double product = this->x * v.x + this->y * v.y;
  return product;
}

__device__ Vector2 Vector2::operator/(const double &d) {
  Vector2 vec;
  vec.set(this->x / d, this->y / d);
  return vec;
}

__device__ Vector2 operator*(const double &d, const Vector2 &v) {
  Vector2 vec;
  vec.set(v.x * d, v.y * d);
  return vec;
}

__device__ void Vector3::clear() {
  this->x = 0.0;
  this->y = 0.0;
  this->z = 0.0;
}

__device__ void Vector3::set(double x, double y, double z) {
  this->x = x;
  this->y = y;
  this->z = z;
}

__device__ double Vector3::norm() {
  return sqrt(x * x + y * y + z * z);
}

__device__ Vector3 &Vector3::operator=(const Vector3 &v) {
  this->x = v.x;
  this->y = v.y;
  this->z = v.z;
  return *this;
}

__device__ Vector3 Vector3::operator+(const Vector3 &v) {
  Vector3 vec;
  vec.set(this->x + v.x, this->y + v.y, this->z + v.z);
  return vec;
}

__device__ Vector3 Vector3::operator-(const Vector3 &v) {
  Vector3 vec;
  vec.set(this->x - v.x, this->y - v.y, this->z - v.z);
  return vec;
}

__device__ Vector3 Vector3::operator*(const double &d) {
  Vector3 vec;
  vec.set(this->x * d, this->y * d, this->z * d);
  return vec;
}

__device__ double Vector3::operator*(const Vector3 &v) {
  double product = this->x * v.x + this->y * v.y + this->z * v.z;
  return product;
}

__device__ Vector3 Vector3::operator/(const double &d) {
  Vector3 vec;
  vec.set(this->x / d, this->y / d, this->z / d);
  return vec;
}

__device__ Vector3 operator*(const double &d, const Vector3 &v) {
  Vector3 vec;
  vec.set(v.x * d, v.y * d, v.z * d);
  return vec;
}

} // namespace Cuda