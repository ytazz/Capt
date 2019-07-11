#ifndef __POLYGON_CUH__
#define __POLYGON_CUH__

#include "vector.cuh"
#include <string>
#include <vector>

namespace GPGPU {

class Polygon {
public:
  Polygon();
  ~Polygon();

  // Set vertices (x, y)
  void setVertex(Vector2 vertex);
  void setVertex(std::vector<Vector2> vertex);

  // Get setted vertices (x, y)
  std::vector<Vector2> getVertex();

  // Find the vertices that make up the convex hull
  // by using gift wrapping algorithm
  std::vector<Vector2> getConvexHull();

  // Find closest point from a point to polygon
  Vector2 getClosestPoint(Vector2 point, std::vector<Vector2> vertex);

  // Determine whether a point (x,y) is within a polygon
  // by using the sign of cross product
  // Especially, ICP and supporting polygon
  // in  -> true
  // out -> false
  bool inPolygon(Vector2 point, std::vector<Vector2> vertex_);

  // Output to the console
  // all vertices
  void printVertex(std::string str);
  void printVertex();
  // the vertices that make up the convex hull
  void printConvex(std::string str);
  void printConvex();

  void clear();

private:
  std::vector<Vector2> vertex;
  std::vector<Vector2> vertex_convex;
};

} // namespace GPGPU

#endif // __POLYGON_CUH__
