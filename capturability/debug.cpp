#include "model.h"
#include "param.h"
#include <chrono>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Model model("data/valkyrie.xml");

  Param param("data/valkyrie_xy.xml");
  param.print();

  return 0;
}