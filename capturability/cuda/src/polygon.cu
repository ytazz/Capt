#include "polygon.cuh"

__device__ Polygon::Polygon() {}

__device__ Polygon::~Polygon() {}

__device__ int Polygon::size(Vector2 *array) {
  int size = (sizeof(array) / sizeof(array[0]));
  return size;
}

__device__ void Polygon::getConvexHull(Vector2 *vertex, Vector2 *convex) {
  Vector2 tmp;
  bool flag_continue = true;
  while (flag_continue) {
    flag_continue = false;
    for (int i = 0; i < size(vertex) - 1; i++) {
      if (((Vector2)vertex[i + 1]).y() < ((Vector2)vertex[i]).y()) {
        tmp = (Vector2)vertex[i];
        (Vector2) vertex[i] = (Vector2)vertex[i + 1];
        (Vector2) vertex[i + 1] = tmp;
        flag_continue = true;
      }
    }
  }

  bool *in_convex = new bool[size(vertex)];
  for (int i = 0; i < size(vertex); i++) {
    in_convex[i] = false;
  }

  int convex_size = 0;
  Vector2 *convex_ = new Vector2[size(vertex)];
  convex_[convex_size] = vertex[0];
  in_convex[0] = true;
  flag_continue = true;
  int back = 0;
  while (flag_continue) {
    flag_continue = false;
    for (int i = 0; i < size(vertex); i++) {
      int product = 0;
      if (!in_convex[i]) {
        product = 1;
        for (int j = 0; j < size(vertex); j++) {
          if (i != j && !in_convex[i]) {
            if (((Vector2)vertex[i] - (Vector2)vertex[back]) %
                    ((Vector2)vertex[j] - (Vector2)vertex[i]) <
                0.0) {
              product *= 0;
            }
          }
        }
      }
      if (product) {
        if (!in_convex[i]) {
          convex_size++;
          convex_[convex_size] = vertex[i];
          in_convex[i] = true;
          flag_continue = true;
          back = i;
        }
        break;
      }
    }
  }
  convex_size++;
  convex_[convex_size] = vertex[0];

  convex = new Vector2[convex_size];
  for (int i = 0; i < convex_size; i++) {
    convex[i] = convex_[i];
  }
}

__device__ Vector2 Polygon::getClosestPoint(Vector2 point, Vector2 *vertex) {
  Vector2 closest;
  Vector2 v1, v2, v3, v4; // vector
  Vector2 n1, n2;         // normal vector

  if (inPolygon(point, vertex)) {
    closest = point;
  } else {
    for (int i = 0; i < size(vertex) - 1; i++) {
      //最近点が角にあるとき
      if (i == 0) {
        n1 = ((Vector2)vertex[1] - (Vector2)vertex[i]).normal();
        n2 = ((Vector2)vertex[i] - (Vector2)vertex[size(vertex) - 2]).normal();
      } else {
        n1 = ((Vector2)vertex[i + 1] - (Vector2)vertex[i]).normal();
        n2 = ((Vector2)vertex[i] - (Vector2)vertex[i - 1]).normal();
      }
      v1 = (Vector2)point - (Vector2)vertex[i];
      if ((n1 % v1) < 0 && (n2 % v1) > 0) {
        closest = vertex[i];
      }
      // 最近点が辺にあるとき
      n1 = ((Vector2)vertex[i + 1] - (Vector2)vertex[i]).normal();
      v1 = point - (Vector2)vertex[i];
      v2 = (Vector2)vertex[i + 1] - (Vector2)vertex[i];
      v3 = point - (Vector2)vertex[i + 1];
      v4 = (Vector2)vertex[i] - (Vector2)vertex[i + 1];
      if ((n1 % v1) > 0 && (v2 % v1) < 0 && (n1 % v3) < 0 && (v4 % v3) > 0) {
        float k = v1 * v2 / (v2.norm() * v2.norm());
        closest = (Vector2)vertex[i] + k * v2;
      }
    }
  }

  return closest;
}

__device__ bool Polygon::inPolygon(Vector2 point, Vector2 *vertex) {
  bool flag = false;
  double product = 0.0;
  int sign = 0, on_line = 0;
  const float epsilon = 0.00001;

  for (size_t i = 0; i < size(vertex) - 1; i++) {
    product = (point - (Vector2)vertex[i]) %
              ((Vector2)vertex[i + 1] - (Vector2)vertex[i]);
    if (-epsilon <= product && product <= epsilon) {
      on_line += 1;
    } else if (product > 0) {
      sign += 1;
    } else if (product < 0) {
      sign -= 1;
    }
  }

  if (sign == int(size(vertex) - 1 - on_line) ||
      sign == -int(size(vertex) - 1 - on_line)) {
    flag = true;
  }

  return flag;
}