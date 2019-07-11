#ifndef __VECTOR_CUH__
#define __VECTOR_CUH__

#include <math.h>
#include <stdio.h>
#include <string>

namespace GPGPU {

typedef struct Vector2 {

  __device__ Vector2(); // constructor
  __device__ ~Vector2();

  __device__ void clear();

  __device__ void setPolar(float radius, float theta);
  __device__ void setCartesian(float x, float y);

  float r, th;
  float x, y;
  __device__ float norm();

  __device__ Vector2 normal();

  __device__ void operator=(const Vector2 &v);

  __device__ Vector2 operator+(const Vector2 &v);
  __device__ Vector2 operator-(const Vector2 &v) const;
  __device__ float operator%(const Vector2 &v);
  __device__ Vector2 operator*(const float &d);
  __device__ float operator*(const Vector2 &v);
  __device__ Vector2 operator/(const float &d);

public:
  __device__ void cartesianToPolar();
  __device__ void polarToCartesian();

} vec2_t;

__device__ Vector2 operator*(const float &d, const Vector2 &v);

} // namespace GPGPU

#endif // __VECTOR_CUH__
