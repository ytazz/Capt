#ifndef __CR_PLOT_H__
#define __CR_PLOT_H__

#include "gnuplot.h"

#include "analysis_cpu.h"
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

  // 踏み出し可能領域を設定
  void setFootRegion();

  void setFoot(State state);
  void setCaptureRegion(State state);
  void setZerostep(State state);

  void initCaptureMap();
  void setCaptureMap(double x, double y, int n_step);

  void plot();

  double omega;

private:
  std::string str(double val);
  std::string str(int val);

  vec2_t cartesianToGraph(vec2_t point);
  vec2_t polarToGraph(vec2_t point);

  Grid grid;

  double x_min, x_max, x_step;
  double y_min, y_max, y_step;
  int    x_num;
  int    y_num;
  int    c_num; // number of color

  std::vector<std::vector<int> > capture_map;
};
}

#endif // __CR_PLOT_H__