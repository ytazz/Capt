#ifndef __GNUPLOT_H__
#define __GNUPLOT_H__

#include <iostream>
#include <map>
#include <stdio.h>
#include <string>

namespace Capt {

class Gnuplot {
public:
  Gnuplot();
  ~Gnuplot();

  void operator()(const std::string &command);

  FILE                      *gp;
  std::map<int, std::string> color;
};
}

#endif // __GNUPLOT_H__