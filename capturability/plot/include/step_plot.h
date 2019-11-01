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
#include <map>

namespace Capt {

class StepPlot : public Gnuplot {
  Gnuplot p;

public:
  StepPlot(Model model, Param param);
  ~StepPlot();

  // 出力ファイル形式を選択(.gif .eps .svg)
  void setOutput(std::string type);

  // 現在の足配置を設定

  // 足配置を設定
  void setFootR(vec2_t foot_r);
  void setFootR(arr2_t foot_r);
  void setFootL(vec2_t foot_l);
  void setFootL(arr2_t foot_l);

  // 現在のICP位置を設定
  void setIcp(vec2_t icp);
  void setIcp(arr2_t icp);

  void plot();

private:
  Model model;
  Param param;
  Grid  grid;

  std::string str(double val);
  std::string str(int val);

  vec2_t cartesianToGraph(vec2_t point);
  vec2_t cartesianToGraph(double x, double y);

  double x_min, x_max, x_step;
  double y_min, y_max, y_step;
  int    x_num;
  int    y_num;

  std::vector<vec2_t> footstep_r;
  std::vector<vec2_t> footstep_l;
  std::vector<vec2_t> icp;
};
}

#endif // __CR_PLOT_H__