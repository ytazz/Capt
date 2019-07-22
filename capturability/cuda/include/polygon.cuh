#ifndef __POLYGON_CUH__
#define __POLYGON_CUH__

#include "vector.cuh"
#include <string>
#include <vector>

class Polygon {
public:
  __device__ Polygon();
  __device__ ~Polygon();

  __device__ int size(Vector2 *array);

  // Find the vertices that make up the convex hull
  // by using gift wrapping algorithm
  __device__ void getConvexHull(Vector2 *vertex, Vector2 *convex);

  // Find closest point from a point to polygon
  __device__ Vector2 getClosestPoint(Vector2 point, Vector2 *vertex);

  // Determine whether a point (x,y) is within a polygon
  // by using the sign of cross product
  // Especially, ICP and supporting polygon
  // in  -> true
  // out -> false
  __device__ bool inPolygon(Vector2 point, Vector2 *vertex);
};

#endif // __POLYGON_CUH__