#include "graph.h"
#include "loader.h"
#include "model.h"
#include <iostream>

using namespace std;
using namespace CA;

int main(int argc, char **argv) {
  Model model("nao.xml");
  model.parse();
  model.print();

  Graph graph("graph.xml");
  graph.parse();
  graph.print();

  std::vector<Vector2> v;
  v=model.getVec("link", "foot_l");
  for (size_t i = 0; i < v.size(); i++) {
    printf("%lf, %lf\n", v[i].x, v[i].y);
  }
}
