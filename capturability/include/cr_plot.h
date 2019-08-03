#ifndef __CR_PLOT_H__
#define __CR_PLOT_H__

#include "gnuplot.h"

#include "capturability.h"
#include "grid.h"
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

class CRPlot : public Gnuplot {
  Gnuplot p;
  Model model;
  Param param;
  Capturability capturability;

public:
  CRPlot(Model model, Param param);
  ~CRPlot();

  void setInput(std::string file_name);
  void setOutput(std::string type);

  void plotCaptureRegion();
  void plotCaptureIcp();
};
}

#endif // __CR_PLOT_H__
