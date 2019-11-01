#include "model.h"
#include <chrono>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Model model("data/valkyrie.xml");
  model.print();
  // Param param("data/valkyrie_xy.xml");

  return 0;
}