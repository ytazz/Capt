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
  Analysis(Model *model, Grid *grid);
  ~Analysis();

  // 解析用変数を初期化し、初期値を代入する
  void initState();
  void initInput();
  void initTrans();
  void initBasin();
  void initNstep();

  // 解析前の状態遷移や0-step、copを計算
  void calcBasin();
  void calcTrans();

  // 解析実行
  void exe();

  // 解析結果をファイルに保存
  void saveTrans(std::string file_name, bool header = false);
  void saveBasin(std::string file_name, bool header = false);
  void saveNstep(std::string file_name, bool header = false);

private:
  bool exe(const int n);

  Model *model;
  Grid  *grid;

  State *state;
  Input *input;
  int   *trans;
  int   *basin;
  int   *nstep;

  const int state_num;
  const int input_num;
  const int grid_num;

  const double epsilon;

  // 解析パラメータ
  double foot_vel_max;
  double step_time_min;

  // 踏み出しできないindexのlist
  std::vector<int> impossible;

  // 解析結果の最大踏み出し歩数
  int max_step;
};

} // namespace Capt

#endif // __ANALYSIS_CPU_H__