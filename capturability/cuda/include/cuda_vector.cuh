#ifndef __CUDA_VECTOR_CUH__
#define __CUDA_VECTOR_CUH__

#include <math.h>
#include <stdio.h>
#include <string>

typedef struct CudaVector2 {
  __device__ void clear();

  __device__ void setPolar(float radius, float theta);
  __device__ void setCartesian(float x, float y);

  __device__ float x();
  __device__ float y();
  __device__ float r();
  __device__ float th();
  __device__ float norm();

  __device__ CudaVector2 normal();

  __device__ CudaVector2 &operator=(const CudaVector2 &v);
  __device__ CudaVector2 operator+(const CudaVector2 &v);
  __device__ CudaVector2 operator-(const CudaVector2 &v);
  __device__ float operator%(const CudaVector2 &v);
  __device__ CudaVector2 operator*(const float &d);
  __device__ float operator*(const CudaVector2 &v);
  __device__ CudaVector2 operator/(const float &d);

public:
  __device__ void cartesianToPolar();
  __device__ void polarToCartesian();

  float r_, th_;
  float x_, y_;

} vec2_t;

__device__ CudaVector2 operator*(const float &d, const CudaVector2 &v);

#endif // __CUDA_VECTOR_CUH__
