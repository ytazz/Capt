#ifndef __ANALYSIS_CPU_H__
#define __ANALYSIS_CPU_H__

#define enableDoubleSupport false

#include "grid.h"
#include "input.h"
#include "model.h"
#include "param.h"
#include "pendulum.h"
#include "state.h"
#include "swing.h"
#include "base.h"
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
  void initInput();
  void initTrans();
  void initBasin();
  void initNstep();

  // calculate 0-step viable-capture basin
  void calcBasin();
  // calculate state transition
  void calcTrans();

  // execute capturability based analysis
  void exe();

  // save results
  void saveTrans(std::string file_name, bool header = false);
  void saveBasin(std::string file_name, bool header = false);
  void saveNstep(std::string file_name, int n, bool header = false);

private:
  // execute capturability based analysis of n-step capture input
  bool exe(const int n);

  Model *model;
  Param *param;
  Grid  *grid;
  Swing *swing;

  State *state;
  Input *input;
  int   *trans;
  int   *basin;
  int   *nstep;

  const int state_num;
  const int input_num;
  const int grid_num;

  // 解析パラメータ
  double v_max;
  // double step_time_min;
  double z_max;

  // 踏み出しできない領域
  double exc_x_min, exc_x_max;
  double exc_y_min, exc_y_max;

  // 解析結果の最大踏み出し歩数
  int max_step;
};

} // namespace Capt

#endif // __ANALYSIS_CPU_H__