#ifndef __GNUPLOT_H__
#define __GNUPLOT_H__

#include <iostream>
#include <map>
#include <stdio.h>
#include <string>

using namespace std;

namespace CA {

class Gnuplot {
public:
  Gnuplot();
  ~Gnuplot();
  void operator()(const string &command);
  FILE *gp;
  map<int, string> int_color;
};
}

#endif // __GNUPLOT_H__
