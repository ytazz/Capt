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

namespace Capt {

class CRPlot : public Gnuplot {
  Gnuplot p;

public:
  CRPlot(Model *model, Param *param);
  ~CRPlot();

  // 出力ファイル形式を選択(.gif .eps .svg)
  void setOutput(std::string type);

  // 踏み出し可能領域を設定
  void setFootRegion();

  // 現在の状態を設定
  void setState(State state);
  void setIcp(vec2_t icp);
  void setSwf(vec2_t swf);

  // Capture Regionのデータを格納するCapture Map
  void setCaptureInput(Input in, int nstep);

  void plot();

private:
  Model *model;
  Param *param;
  Grid  *grid;

  std::string str(float val);
  std::string str(int val);

  vec2_t cartesianToGraph(vec2_t point);
  vec2_t cartesianToGraph(float x, float y);

  float  swf_x_min, swf_x_max;
  float  swf_y_min, swf_y_max;

  float  cop_x_min, cop_x_max;
  float  cop_y_min, cop_y_max;

  float  exc_x_min, exc_x_max;
  float  exc_y_min, exc_y_max;

  float  grid_x_min, grid_x_max;
  float  grid_y_min, grid_y_max;

  int c_num;    // number of color

  std::vector< std::pair<Input, int> > cap_input;

};
}

#endif // __CR_PLOT_H__
