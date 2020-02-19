#ifndef __CUDA_VECTOR_CUH__
#define __CUDA_VECTOR_CUH__

#include <math.h>
#include <stdio.h>
#include <string>

namespace Cuda {

typedef struct Vector2 {
  __device__ void clear();

  __device__ void set(double x, double y);

  __device__ double norm();

  __device__ Vector2 normal();

  __device__ Vector2 &operator=(const Vector2 &v);
  __device__ Vector2 operator +(const Vector2 &v);
  __device__ Vector2 operator -(const Vector2 &v);
  __device__ double operator  %(const Vector2 &v);
  __device__ Vector2 operator *(const double &d);
  __device__ double operator  *(const Vector2 &v);
  __device__ Vector2 operator /(const double &d);

  double x, y;

} vec2_t;

__device__ Vector2 operator*(const double &d, const Vector2 &v);

} // namespace Cuda

#endif // __CUDA_VECTOR_CUH__