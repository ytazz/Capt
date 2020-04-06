#ifndef __CUDA_VECTOR_CUH__
#define __CUDA_VECTOR_CUH__

#include <math.h>
#include <stdio.h>
#include <string>

namespace Cuda {

typedef struct Vector2 {
  __device__ void clear();

  __device__ void set(float x, float y);

  __device__ float norm();

  __device__ Vector2 normal();

  __device__ Vector2 &operator=(const Vector2 &v);
  __device__ Vector2 operator +(const Vector2 &v);
  __device__ Vector2 operator -(const Vector2 &v);
  __device__ float   operator  %(const Vector2 &v);
  __device__ Vector2 operator *(const float &d);
  __device__ float   operator  *(const Vector2 &v);
  __device__ Vector2 operator /(const float &d);

  float x, y;

} vec2_t;

__device__ Vector2 operator*(const float &d, const Vector2 &v);

typedef struct Vector3 {
  __device__ void clear();

  __device__ void set(float x, float y, float z);

  __device__ float norm();

  __device__ Vector3 &operator=(const Vector3 &v);
  __device__ Vector3 operator +(const Vector3 &v);
  __device__ Vector3 operator -(const Vector3 &v);
  // __device__ float operator  %(const Vector3 &v);
  __device__ Vector3 operator *(const float   &d);
  __device__ float   operator  *(const Vector3 &v);
  __device__ Vector3 operator /(const float &d);

  float x, y, z;

} vec3_t;

__device__ Vector3 operator*(const float &d, const Vector3 &v);

} // namespace Cuda

#endif // __CUDA_VECTOR_CUH__