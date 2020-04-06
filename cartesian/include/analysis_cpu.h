#ifndef __ANALYSIS_CPU_H__
#define __ANALYSIS_CPU_H__

//#define enableDoubleSupport false

#include "grid.h"
#include "input.h"
#include "model.h"
#include "param.h"
#include "pendulum.h"
#include "state.h"
#include "swing.h"
#include "base.h"
#include "capturability.h"
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <vector>

namespace Capt {

class Analysis {
public:
  Analysis(Model *model, Param *param, Grid *grid);
  ~Analysis();

  // 解析用変数を初期化し、初期値を代入する
  void initState();
  void initBasin();
  void initNstep();

  // calculate 0-step viable-capture basin
  void calcBasin();

  // execute capturability based analysis
  void exe();

private:
  // execute capturability based analysis of n-step capture input
  bool exe(const int n);

  Model *model;
  Param *param;
  Grid  *grid;
  Swing *swing;
  Capturability* capturability;


  // 解析パラメータ
  float v_max;
  // double step_time_min;
  float z_max;

};

} // namespace Capt

#endif // __ANALYSIS_CPU_H__