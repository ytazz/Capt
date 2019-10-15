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
  Gnuplot p;
  Model   model;
  Param   param;
  // Capturability capturability;

public:
  CRPlot(Model model, Param param);
  ~CRPlot();

  void setOutput(std::string type);

  // void setCaptureRegion(State state);
  void setCaptureRegion();

  void setFoot(State state);
  void setZerostep(State state);

  void initCaptureMap();
  void setCaptureMap(double x, double y, int n_step);

  void plot();

  double omega;

private:
  std::string str(double val);
  std::string str(int val);

  Grid grid;

  double x_min, x_max, x_step;
  double y_min, y_max, y_step;
  int    x_num;
  int    y_num;

  std::vector<std::vector<int> > capture_map;
};
}

#endif // __CR_PLOT_H__