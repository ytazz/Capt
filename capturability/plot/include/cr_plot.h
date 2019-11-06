#ifndef __CR_PLOT_H__
#define __CR_PLOT_H__

#include "gnuplot.h"

#include "analysis_cpu.h"
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
  CRPlot(Model *model, Param *param, Grid *grid);
  ~CRPlot();

  // 出力ファイル形式を選択(.gif .eps .svg)
  void setOutput(std::string type);

  // 踏み出し可能領域を設定
  void setFootRegion();

  // 現在の足配置を設定
  void setFoot(vec2_t swf);

  // 現在のICP位置を設定
  void setIcp(vec2_t icp);

  // Capture Regionのデータを格納するCapture Map
  void initCaptureMap();
  void setCaptureMap(double x, double y, int n_step);

  void plot();

private:
  Model *model;
  Param *param;
  Grid  *grid;

  std::string str(double val);
  std::string str(int val);

  vec2_t cartesianToGraph(vec2_t point);
  vec2_t cartesianToGraph(double x, double y);

  double x_min, x_max, x_stp;
  double y_min, y_max, y_stp;
  double swf_x_min, swf_x_max;
  double swf_y_min, swf_y_max;

  int x_num;
  int y_num;
  int swf_x_num, swf_y_num;
  int c_num;    // number of color

  std::vector<std::vector<int> > capture_map;
};
}

#endif // __CR_PLOT_H__