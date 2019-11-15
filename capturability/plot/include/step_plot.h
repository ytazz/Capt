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
  StepPlot(Model *model, Param *param, Grid *grid);
  ~StepPlot();

  // 出力ファイル形式を選択(.gif .eps .svg)
  void setOutput(std::string type);

  // 足配置を設定
  void setFootR(vec2_t foot_r);
  void setFootR(arr2_t foot_r);
  void setFootL(vec2_t foot_l);
  void setFootL(arr2_t foot_l);

  // ICP位置を設定
  void setIcp(vec2_t icp);
  void setIcp(arr2_t icp);

  // Cop位置を設定
  void setCop(vec2_t cop);
  void setCop(arr2_t cop);

  void setTransition(std::vector<State> states, std::vector<Input> inputs, Foot suf);

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
  int    x_num;
  int    y_num;

  arr2_t footstep_r;
  arr2_t footstep_l;
  arr2_t icp;
  arr2_t cop;

  std::vector<State> states;
  std::vector<Input> inputs;
};
}

#endif // __CR_PLOT_H__