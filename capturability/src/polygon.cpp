#include "polygon.h"

namespace Capt {

Polygon::Polygon() {
  vertex.clear();
  vertex_convex.clear();
}

Polygon::~Polygon() {}

void Polygon::setVertex(Vector2 vertex) { this->vertex.push_back(vertex); }

void Polygon::setVertex(std::vector<Vector2> vertex) {
  for (size_t i = 0; i < vertex.size() - 1; i++) {
    this->vertex.push_back(vertex[i]);
  }
}

std::vector<Vector2> Polygon::getVertex() { return vertex; }

std::vector<Vector2> Polygon::getConvexHull() {
  if (vertex.size() < 3) {
    printf("Error: setted vertices(%d) are too small (< 3)\n",
           (int)vertex.size());
    exit(EXIT_FAILURE);
  }

  Vector2 tmp;
  bool flag_continue = true;
  while (flag_continue) {
    flag_continue = false;
    for (size_t i = 0; i < vertex.size() - 1; i++) {
      if (vertex[i + 1].y < vertex[i].y) {
        tmp = vertex[i];
        vertex[i] = vertex[i + 1];
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
  in_convex[0] = true;
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
            if ((vertex[i] - vertex[back]) % (vertex[j] - vertex[i]) < 0.0) {
              product *= 0;
            }
          }
        }
      }
      if (product) {
        if (!in_convex[i]) {
          vertex_convex.push_back(vertex[i]);
          in_convex[i] = true;
          flag_continue = true;
          back = i;
        }
        break;
      }
    }
  }
  vertex_convex.push_back(vertex[0]);

  return vertex_convex;
}

Vector2 Polygon::getClosestPoint(Vector2 point, std::vector<Vector2> vertex) {
  Vector2 closest;
  Vector2 v1, v2, v3, v4; // vector
  Vector2 n1, n2;         // normal vector

  if (inPolygon(point, vertex)) {
    closest = point;
  } else {
    for (size_t i = 0; i < vertex.size() - 1; i++) {
      //最近点が角にあるとき
      if (i == 0) {
        n1 = (vertex[1] - vertex[i]).normal();
        n2 = (vertex[i] - vertex[vertex.size() - 2]).normal();
      } else {
        n1 = (vertex[i + 1] - vertex[i]).normal();
        n2 = (vertex[i] - vertex[i - 1]).normal();
      }
      v1 = point - vertex[i];
      if ((n1 % v1) < 0 && (n2 % v1) > 0) {
        closest = vertex[i];
      }
      // 最近点が辺にあるとき
      n1 = (vertex[i + 1] - vertex[i]).normal();
      v1 = point - vertex[i];
      v2 = vertex[i + 1] - vertex[i];
      v3 = point - vertex[i + 1];
      v4 = vertex[i] - vertex[i + 1];
      if ((n1 % v1) > 0 && (v2 % v1) < 0 && (n1 % v3) < 0 && (v4 % v3) > 0) {
        float k = v1 * v2 / (v2.norm() * v2.norm());
        closest = vertex[i] + k * v2;
      }
    }
  }

  return closest;
}

bool Polygon::inPolygon(Vector2 point, std::vector<Vector2> vertex_) {
  bool flag = false;
  double product = 0.0;
  int sign = 0, on_line = 0;
  const float epsilon = 0.00001;

  for (size_t i = 0; i < vertex_.size() - 1; i++) {
    product = (point - vertex_[i]) % (vertex_[i + 1] - vertex_[i]);
    if (-epsilon <= product && product <= epsilon) {
      on_line += 1;
    } else if (product > 0) {
      sign += 1;
    } else if (product < 0) {
      sign -= 1;
    }
  }

  if (sign == int(vertex_.size() - 1 - on_line) ||
      sign == -int(vertex_.size() - 1 - on_line)) {
    flag = true;
  }

  return flag;
}

void Polygon::printVertex(std::string str) {
  for (size_t i = 0; i < vertex.size(); i++) {
    printf("%s %lf, %lf\n", str.c_str(), vertex[i].x, vertex[i].y);
  }
}

void Polygon::printVertex() { printVertex(""); }

void Polygon::printConvex(std::string str) {
  for (size_t i = 0; i < vertex_convex.size(); i++) {
    printf("%s %lf, %lf\n", str.c_str(), vertex_convex[i].x,
           vertex_convex[i].y);
  }
}

void Polygon::printConvex() { printConvex(""); }

void Polygon::clear() {
  vertex.clear();
  vertex_convex.clear();
}
} // namespace Capt
