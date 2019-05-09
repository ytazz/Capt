#include "loader.h"
#include <iostream>

using namespace std;

int main(int argc, char **argv) {
  Loader loader("nao.xml");
  loader.parse();
}
