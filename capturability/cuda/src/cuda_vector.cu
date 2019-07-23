#include "cuda_vector.cuh"

__device__ CudaVector2::CudaVector2() { clear(); }

__device__ void CudaVector2::clear() {
  this->r_ = 0.0;
  this->th_ = 0.0;
  this->x_ = 0.0;
  this->y_ = 0.0;
}

__device__ void CudaVector2::setPolar(float radius, float theta) {
  this->r_ = radius;
  this->th_ = theta;
  polarToCartesian();
}

__device__ void CudaVector2::setCartesian(float x, float y) {
  this->x_ = x;
  this->y_ = y;
  cartesianToPolar();
}

__device__ void CudaVector2::polarToCartesian() {
  this->x_ = this->r_ * cos(this->th_);
  this->y_ = this->r_ * sin(this->th_);
}

__device__ void CudaVector2::cartesianToPolar() {
  this->r_ = norm();
  this->th_ = atan2f(this->y_, this->x_);
  if (this->th_ < 0.0) {
    this->th_ += 2 * M_PI;
  }
}

__device__ float CudaVector2::x() { return this->x_; }

__device__ float CudaVector2::y() { return this->y_; }

__device__ float CudaVector2::r() { return this->r_; }

__device__ float CudaVector2::th() { return this->th_; }

__device__ float CudaVector2::norm() { return sqrt(x_ * x_ + y_ * y_); }

__device__ CudaVector2 CudaVector2::normal() {
  CudaVector2 normal_vector;
  // rotate -90 deg around +z direction
  normal_vector.setCartesian(this->y_, -this->x_);
  return normal_vector;
}

__device__ CudaVector2 &CudaVector2::operator=(const CudaVector2 &v) {
  this->x_ = v.x_;
  this->y_ = v.y_;
  this->r_ = v.r_;
  this->th_ = v.th_;
  return *this;
}

__device__ CudaVector2 CudaVector2::operator+(const CudaVector2 &v) {
  CudaVector2 vec;
  double x, y;
  x = this->x_ + v.x_;
  y = this->y_ + v.y_;
  vec.setCartesian(x, y);
  return vec;
}

__device__ CudaVector2 CudaVector2::operator-(const CudaVector2 &v) {
  CudaVector2 vec;
  double x, y;
  x = this->x_ - v.x_;
  y = this->y_ - v.y_;
  vec.setCartesian(x, y);
  return vec;
}

__device__ float CudaVector2::operator%(const CudaVector2 &v) {
  float product;
  product = this->x_ * v.y_ - this->y_ * v.x_;
  return product;
}

__device__ CudaVector2 CudaVector2::operator*(const float &d) {
  CudaVector2 vec;
  double x, y;
  x = this->x_ * d;
  y = this->y_ * d;
  vec.setCartesian(x, y);
  return vec;
}

__device__ float CudaVector2::operator*(const CudaVector2 &v) {
  float product = 0.0;
  product += this->x_ * v.x_;
  product += this->y_ * v.y_;
  return product;
}

__device__ CudaVector2 CudaVector2::operator/(const float &d) {
  CudaVector2 vec;
  double x, y;
  x = this->x_ / d;
  y = this->y_ / d;
  vec.setCartesian(x, y);
  return vec;
}

__device__ CudaVector2 operator*(const float &d, const CudaVector2 &v) {
  CudaVector2 vec;
  double x, y;
  x = v.x_ * d;
  y = v.y_ * d;
  vec.setCartesian(x, y);
  return vec;
}