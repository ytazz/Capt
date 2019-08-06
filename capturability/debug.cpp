#include "vector.h"
#include <iostream>

using namespace std;
using namespace Capt;

void getConvexHull(Vector2 *vertex, Vector2 *convex) {
  std::cout << "0" << '\n';

  Vector2 tmp;
  bool flag_continue = true;
  while (flag_continue) {
    flag_continue = false;
    std::cout << "loop" << '\n';
    for (int i = 0; i < 22 - 1; i++) {
      if ((vertex[i + 1]).y < (vertex[i]).y) {
        tmp = vertex[i];
        vertex[i] = vertex[i + 1];
        vertex[i + 1] = tmp;
        flag_continue = true;
      }
    }
  }

  std::cout << "1" << '\n';

  bool in_convex[22];
  for (int i = 0; i < 22; i++) {
    in_convex[i] = false;
  }

  std::cout << "2" << '\n';

  int convex_size = 0;
  convex[convex_size] = vertex[0];
  in_convex[0] = true;
  flag_continue = true;
  int back = 0;
  while (flag_continue) {
    flag_continue = false;
    for (int i = 0; i < 22; i++) {
      int product = 0;
      if (!in_convex[i]) {
        product = 1;
        for (int j = 0; j < 22; j++) {
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
          convex[convex_size] = vertex[i];
          in_convex[i] = true;
          flag_continue = true;
          back = i;
        }
        break;
      }
    }
  }

  std::cout << "3" << '\n';

  convex_size++;
  convex[convex_size] = vertex[0];

  std::cout << "4" << '\n';

  for (int i = convex_size; i < 22; i++) {
    convex[i] = vertex[0];
  }

  // std::cout << "5" << '\n';
}

int main(int argc, char const *argv[]) {
  Vector2 foot[22], convex[22];
  double offset_x = 0.0, offset_y = 0.1;

  foot[0].setCartesian(-0.065, -0.025);
  foot[1].setCartesian(-0.055, -0.035);
  foot[2].setCartesian(-0.015, -0.025);
  foot[3].setCartesian(+0.015, -0.035);
  foot[4].setCartesian(+0.035, -0.035);
  foot[5].setCartesian(+0.065, -0.015);
  foot[6].setCartesian(+0.065, +0.025);
  foot[7].setCartesian(+0.035, +0.035);
  foot[8].setCartesian(-0.015, +0.025);
  foot[9].setCartesian(-0.065, +0.025);
  foot[10].setCartesian(-0.065, -0.025);

  foot[11].setCartesian(-0.065 + offset_x, +0.025 + offset_y);
  foot[12].setCartesian(-0.055 + offset_x, +0.035 + offset_y);
  foot[13].setCartesian(-0.015 + offset_x, +0.025 + offset_y);
  foot[14].setCartesian(+0.015 + offset_x, +0.035 + offset_y);
  foot[15].setCartesian(+0.035 + offset_x, +0.035 + offset_y);
  foot[16].setCartesian(+0.065 + offset_x, +0.015 + offset_y);
  foot[17].setCartesian(+0.065 + offset_x, -0.025 + offset_y);
  foot[18].setCartesian(+0.035 + offset_x, -0.035 + offset_y);
  foot[19].setCartesian(-0.015 + offset_x, -0.025 + offset_y);
  foot[20].setCartesian(-0.065 + offset_x, -0.025 + offset_y);
  foot[21].setCartesian(-0.065 + offset_x, +0.025 + offset_y);

  getConvexHull(foot, convex);

  for (int i = 0; i < 22; i++) {
    printf("%d: \t%1.4lf, \t%1.4lf\n", i, foot[i].x, foot[i].y);
  }
  printf("--------------\n");
  for (int i = 0; i < 22; i++) {
    printf("%d: \t%1.4lf, \t%1.4lf\n", i, convex[i].x, convex[i].y);
  }

  return 0;
}
