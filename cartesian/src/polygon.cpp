#include "polygon.h"

namespace Capt {

Polygon::Polygon() {
  vertex.clear();
  vertex_convex.clear();
}

Polygon::~Polygon() {
}

void Polygon::setVertex(vec2_t vertex) {
  this->vertex.push_back(vertex);
}

void Polygon::setVertex(arr2_t vertex) {
  for (size_t i = 0; i < vertex.size() - 1; i++) {
    this->vertex.push_back(vertex[i]);
  }
}

arr2_t Polygon::getVertex() {
  return vertex;
}

arr2_t Polygon::getConvexHull() {
  if (vertex.size() < 3) {
    printf("Error: setted vertices(%d) are too small (< 3)\n",
           (int)vertex.size() );
    exit(EXIT_FAILURE);
  }

  vec2_t tmp;
  bool   flag_continue = true;
  while (flag_continue) {
    flag_continue = false;
    for (size_t i = 0; i < vertex.size() - 1; i++) {
      if (vertex[i + 1].y() < vertex[i].y() ) {
        tmp           = vertex[i];
        vertex[i]     = vertex[i + 1];
        vertex[i + 1] = tmp;
        flag_continue = true;
      }
    }
  }

  std::vector<bool> in_convex;
  for (size_t i = 0; i < vertex.size(); i++) {
    in_convex.push_back(false);
  }

  vertex_convex.push_back(vertex[0]);
  in_convex[0]  = true;
  flag_continue = true;
  int back = 0;
  while (flag_continue) {
    flag_continue = false;
    for (size_t i = 0; i < vertex.size(); i++) {
      int product = 0;
      if (!in_convex[i]) {
        product = 1;
        for (size_t j = 0; j < vertex.size(); j++) {
          if (i != j && !in_convex[i]) {
            if ( cross( ( vertex[i] - vertex[back] ), ( vertex[j] - vertex[i] ) ) < 0.0) {
              product *= 0;
            }
          }
        }
      }
      if (product > 0) {
        if (!in_convex[i]) {
          vertex_convex.push_back(vertex[i]);
          in_convex[i]  = true;
          flag_continue = true;
          back          = i;
        }
        break;
      }
    }
  }
  vertex_convex.push_back(vertex[0]);

  return vertex_convex;
}

vec2_t Polygon::getClosestPoint(vec2_t point, arr2_t vertex) {
  vec2_t closest;
  vec2_t v1, v2, v3, v4; // vector
  vec2_t n1, n2;         // normal vector

  if (inPolygon(point, vertex) ) {
    closest = point;
  } else {
    for (size_t i = 0; i < vertex.size() - 1; i++) {
      //最近点が角にあるとき
      if (i == 0) {
        n1 = normal( vertex[1] - vertex[i] );
        n2 = normal( vertex[i] - vertex[vertex.size() - 2] );
      } else {
        n1 = normal( vertex[i + 1] - vertex[i] );
        n2 = normal( vertex[i] - vertex[i - 1] );
      }
      v1 = point - vertex[i];
      if ( cross( n1, v1 ) < 0 && cross( n2, v1 ) > 0) {
        closest = vertex[i];
      }
      // 最近点が辺にあるとき
      n1 = normal( vertex[i + 1] - vertex[i] );
      v1 = point - vertex[i];
      v2 = vertex[i + 1] - vertex[i];
      v3 = point - vertex[i + 1];
      v4 = vertex[i] - vertex[i + 1];
      if ( cross(n1, v1) > 0 && cross(v2, v1) < 0 && cross(n1, v3 ) < 0 && cross(v4, v3 ) > 0) {
        float k = dot(v1, v2) / ( v2.norm() * v2.norm() );
        closest = vertex[i] + k * v2;
      }
    }
  }

  return closest;
}

bool Polygon::inPolygon(vec2_t point, arr2_t vertex_) {
  float product = 0.0f;
  int   sign    = 0;
  int   on_line = 0;
  const float epsilon = 0.00001f;

  for (size_t i = 0; i < vertex_.size() - 1; i++) {
    product = cross( ( point - vertex_[i] ), ( vertex_[i + 1] - vertex_[i] ) );
    if (-epsilon <= product && product <= epsilon) {
      on_line += 1;
    } else if (product > 0) {
      sign += 1;
    } else if (product < 0) {
      sign -= 1;
    }
  }

  return ( std::abs(sign) + on_line == (int)vertex_.size() - 1 );
}

void Polygon::printVertex(std::string str) {
  for (size_t i = 0; i < vertex.size(); i++) {
    printf("%s %lf, %lf\n", str.c_str(), vertex[i].x(), vertex[i].y() );
  }
}

void Polygon::printVertex() {
  printVertex("");
}

void Polygon::printConvex(std::string str) {
  for (size_t i = 0; i < vertex_convex.size(); i++) {
    printf("%s %lf, %lf\n", str.c_str(), vertex_convex[i].x(),
           vertex_convex[i].y() );
  }
}

void Polygon::printConvex() {
  printConvex("");
}

void Polygon::clear() {
  vertex.clear();
  vertex_convex.clear();
}
} // namespace Capt
