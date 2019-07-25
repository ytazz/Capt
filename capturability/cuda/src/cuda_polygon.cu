#include "cuda_polygon.cuh"

__device__ CudaPolygon::CudaPolygon() {}

__device__ CudaPolygon::~CudaPolygon() {}

__device__ int CudaPolygon::size(CudaVector2 *array) {
  int size = (sizeof(array) / sizeof(array[0]));
  return size;
}

__device__ void CudaPolygon::getConvexHull(CudaVector2 *vertex,
                                           CudaVector2 *convex) {
  CudaVector2 tmp;
  bool flag_continue = true;
  while (flag_continue) {
    flag_continue = false;
    for (int i = 0; i < 22 - 1; i++) {
      if ((vertex[i + 1]).y() < (vertex[i]).y()) {
        tmp = vertex[i];
        vertex[i] = vertex[i + 1];
        vertex[i + 1] = tmp;
        flag_continue = true;
      }
    }
  }

  bool in_convex[22];
  for (int i = 0; i < 22; i++) {
    in_convex[i] = false;
  }

  int convex_size = 0;
  convex[convex_size] = vertex[0];
  in_convex[0] = true;
  flag_continue = true;
  int back = 0;
  // while (flag_continue) {
  flag_continue = false;
  for (int i = 0; i < 22; i++) {
    int product = 0;
    if (!in_convex[i]) {
      product = 1;
      for (int j = 0; j < 22; j++) {
        if (i != j && !in_convex[i]) {
          if ((vertex[i] - vertex[back]) % (vertex[j] - vertex[i]) < 0.0) {
            product *= 0;
          }
        }
      }
    }
    if (product) {
      if (!in_convex[i]) {
        convex_size++;
        convex[convex_size] = vertex[i];
        in_convex[i] = true;
        flag_continue = true;
        back = i;
      }
      break;
    }
  }
  // }
  convex_size++;
  convex[convex_size] = vertex[0];

  for (int i = convex_size; i < size(vertex); i++) {
    convex[i] = vertex[0];
  }
  // for (int i = 0; i < 22; i++) {
  //   convex[i] = vertex[i];
  // }
}

__device__ CudaVector2 CudaPolygon::getClosestPoint(CudaVector2 point,
                                                    CudaVector2 *vertex) {
  CudaVector2 closest;
  CudaVector2 v1, v2, v3, v4; // vector
  CudaVector2 n1, n2;         // normal vector

  if (inPolygon(point, vertex)) {
    closest = point;
  } else {
    for (int i = 0; i < size(vertex) - 1; i++) {
      //最近点が角にあるとき
      if (i == 0) {
        n1 = ((CudaVector2)vertex[1] - (CudaVector2)vertex[i]).normal();
        n2 = ((CudaVector2)vertex[i] - (CudaVector2)vertex[size(vertex) - 2])
                 .normal();
      } else {
        n1 = ((CudaVector2)vertex[i + 1] - (CudaVector2)vertex[i]).normal();
        n2 = ((CudaVector2)vertex[i] - (CudaVector2)vertex[i - 1]).normal();
      }
      v1 = (CudaVector2)point - (CudaVector2)vertex[i];
      if ((n1 % v1) < 0 && (n2 % v1) > 0) {
        closest = vertex[i];
      }
      // 最近点が辺にあるとき
      n1 = ((CudaVector2)vertex[i + 1] - (CudaVector2)vertex[i]).normal();
      v1 = point - (CudaVector2)vertex[i];
      v2 = (CudaVector2)vertex[i + 1] - (CudaVector2)vertex[i];
      v3 = point - (CudaVector2)vertex[i + 1];
      v4 = (CudaVector2)vertex[i] - (CudaVector2)vertex[i + 1];
      if ((n1 % v1) > 0 && (v2 % v1) < 0 && (n1 % v3) < 0 && (v4 % v3) > 0) {
        float k = v1 * v2 / (v2.norm() * v2.norm());
        closest = (CudaVector2)vertex[i] + k * v2;
      }
    }
  }

  return closest;
}

__device__ bool CudaPolygon::inPolygon(CudaVector2 point, CudaVector2 *vertex) {
  bool flag = false;
  double product = 0.0;
  int sign = 0, on_line = 0;
  const float epsilon = 0.00001;

  for (size_t i = 0; i < size(vertex) - 1; i++) {
    product = (point - (CudaVector2)vertex[i]) %
              ((CudaVector2)vertex[i + 1] - (CudaVector2)vertex[i]);
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