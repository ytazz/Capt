#ifndef __CUDA_POLYGON_CUH__
#define __CUDA_POLYGON_CUH__

#include "cuda_vector.cuh"
#include <string>
#include <vector>

class CudaPolygon {
public:
  __device__ CudaPolygon();
  __device__ ~CudaPolygon();

  __device__ int size(CudaVector2 *array);

  // Find the vertices that make up the convex hull
  // by using gift wrapping algorithm
  __device__ void getConvexHull(CudaVector2 *vertex, CudaVector2 *convex);

  // Find closest point from a point to polygon
  __device__ CudaVector2 getClosestPoint(CudaVector2 point,
                                         CudaVector2 *vertex);

  // Determine whether a point (x,y) is within a polygon
  // by using the sign of cross product
  // Especially, ICP and supporting polygon
  // in  -> true
  // out -> false
  __device__ bool inPolygon(CudaVector2 point, CudaVector2 *vertex);
};

#endif // __CUDA_POLYGON_CUH__