#pragma once

#include "base.h"
#include <string>
#include <vector>

namespace Capt {

class Polygon {
public:
  Polygon();
  ~Polygon();

  // Set vertices (x, y)
  void setVertex(vec2_t vertex);
  void setVertex(arr2_t vertex);

  // Get setted vertices (x, y)
  arr2_t getVertex();

  // Find the vertices that make up the convex hull
  // by using gift wrapping algorithm
  arr2_t getConvexHull();

  // Find closest point from a point to polygon
  vec2_t getClosestPoint(vec2_t point, arr2_t vertex);

  // Determine whether a point (x,y) is within a polygon
  // by using the sign of cross product
  // Especially, ICP and supporting polygon
  // in  -> true
  // out -> false
  bool inPolygon(vec2_t point, arr2_t vertex_);

  // Output to the console
  // all vertices
  void printVertex(std::string str);
  void printVertex();
  // the vertices that make up the convex hull
  void printConvex(std::string str);
  void printConvex();

  void clear();

private:
  arr2_t vertex;
  arr2_t vertex_convex;
};

} // namespace Capt
