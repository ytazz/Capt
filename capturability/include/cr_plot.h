#ifndef __CR_PLOT_H__
#define __CR_PLOT_H__

#include "gnuplot.h"

#include "capturability.h"
#include "input.h"
#include "param.h"
#include "state.h"

#include <algorithm>
#include <ctime>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

namespace CA {

class CRPlot {
  Gnuplot p;
  Model model;
  Param param;

public:
  CRPlot(Model model, Param param);
  ~CRPlot();

  void plot(State state, std::vector<CaptureSet> region);
  void plot();
};
}

#endif // __CR_PLOT_H__
