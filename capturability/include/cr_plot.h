#ifndef __CR_PLOT_H__
#define __CR_PLOT_H__

#include "gnuplot.h"

#include "capturability.h"
#include "friction_filter.h"
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

namespace Capt {

class CRPlot : public Gnuplot {
  Gnuplot       p;
  Model         model;
  Param         param;
  Capturability capturability;

public:
  CRPlot(Model model, Param param);
  ~CRPlot();

  void setInput(std::string file_name, DataType type);
  void setOutput(std::string type);

  void animCaptureRegion(State state);
  void plotCaptureRegion(State state);
  void plotCaptureIcp(State state);

  double omega;

  vec2_t com;
  double mu;
};
}

#endif // __CR_PLOT_H__
