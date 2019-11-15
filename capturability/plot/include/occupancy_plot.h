#ifndef __MAP_PLOT_H__
#define __MAP_PLOT_H__

#include "gnuplot.h"

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

enum OccupancyType {
  NONE,
  EMPTY,
  OBSTACLE,
  OPEN,
  CLOSED
};

class OccupancyPlot : public Gnuplot {
  Gnuplot p;

public:
  OccupancyPlot(Param *param);
  ~OccupancyPlot();

  // 出力ファイル形式を選択(.gif .eps .svg)
  void setOutput(std::string type);

  // 障害物位置を設定
  // Open/Close Nodeを設定
  // 決定された着地点を設定
  void setOccupancy(double x, double y, OccupancyType type);
  void setOccupancy(vec2_t pos, OccupancyType type);

  // GridMapのデータを格納するOccupancy
  void initOccupancy();

  void plot();

private:
  Param *param;

  std::string str(double val);
  std::string str(int val);

  vec2_t cartesianToGraph(vec2_t point);
  vec2_t cartesianToGraph(double x, double y);

  double x_min, x_max, x_stp;
  double y_min, y_max, y_stp;

  int x_num;
  int y_num;

  std::vector<std::vector<int> > occupancy;
};

}

#endif // __MAP_PLOT_H__