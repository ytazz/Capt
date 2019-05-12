#include "loader.h"
#include "model.h"
#include <iostream>

using namespace std;
using namespace CA;

int main(int argc, char **argv) {
  Model model("nao.xml");
  model.parse();
  model.print();
}
