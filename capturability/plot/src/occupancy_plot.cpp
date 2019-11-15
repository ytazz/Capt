#include "occupancy_plot.h"

using namespace std;

namespace Capt {

OccupancyPlot::OccupancyPlot(Param *param)
  : param(param){
  // ファイル形式の確認
  // p("unset key");
  p("set encoding utf8");
  // p("set terminal qt size 600,600");

  param->read(&x_min, "map_x_min");
  param->read(&x_max, "map_x_max");
  param->read(&x_stp, "map_x_stp");
  param->read(&x_num, "map_x_num");
  param->read(&y_min, "map_y_min");
  param->read(&y_max, "map_y_max");
  param->read(&y_stp, "map_y_stp");
  param->read(&y_num, "map_y_num");

  // グラフサイズ設定
  p("set size square");

  // 軸ラベル設定
  p("set xlabel 'y [m]'");
  p("set ylabel 'x [m]'");
  p("set xlabel font \"Arial,15\"");
  p("set ylabel font \"Arial,15\"");
  p("set tics   font \"Arial,15\"");
  p("set key    font \"Arial,15\"");

  // 座標軸の目盛り設定
  p("set xtics 1");
  p("set ytics 1");
  p("set mxtics 2");
  p("set mytics 2");
  p("set xtics scale 0,0.001");
  p("set ytics scale 0,0.001");

  // xy軸の目盛りの値を再設定
  std::string x_tics, y_tics;
  for (int i = -1; i <= x_num; i++) {
    x_tics += "\"\" " + str(i);
    if (i != x_num)
      x_tics += ", ";
  }
  for (int i = -1; i <= y_num; i++) {
    y_tics += "\"\" " + str(i);
    if (i != y_num)
      y_tics += ", ";
  }
  fprintf(p.gp, "set xtics add (%s)\n", x_tics.c_str() );
  fprintf(p.gp, "set ytics add (%s)\n", y_tics.c_str() );
  fprintf(p.gp, "set xtics add (\"%1.1lf\" 0, \"%1.1lf\" %d)\n", y_max, y_min, ( y_num - 1 ) );
  fprintf(p.gp, "set ytics add (\"%1.1lf\" 0, \"%1.1lf\" %d)\n", x_min, x_max, ( x_num - 1 ) );

  // 色の設定
  // 0: white
  // 1: black (obstacle)
  // 2: blue  (open)
  // 3: red   (footstep path)
  fprintf(p.gp, "set palette defined ( 0 '#ffffff', 1 '#000000', 2 '#0097ff', 3 '#ff0000')\n");
  fprintf(p.gp, "set cbrange[0:3]\n");
  fprintf(p.gp, "set palette maxcolors 4\n");
  fprintf(p.gp, "unset colorbox\n");

  initOccupancy();
}

OccupancyPlot::~OccupancyPlot() {
}

std::string OccupancyPlot::str(double val){
  return std::to_string(val);
}

std::string OccupancyPlot::str(int val){
  return std::to_string(val);
}

void OccupancyPlot::setOutput(std::string type) {
  if (type == "gif") {
    p("set terminal gif animate optimize delay 50 size 600,900");
    p("set output 'plot.gif'");
  }
  if (type == "svg") {
    p("set terminal svg");
    p("set output 'plot.svg'");
  }
  if (type == "eps") {
    p("set terminal postscript eps enhanced");
    p("set output 'plot.eps'");
  }
}

void OccupancyPlot::initOccupancy(){
  occupancy.clear();
  occupancy.resize(x_num);
  for (int i = 0; i < x_num; i++) {
    occupancy[i].resize(y_num);
    for (int j = 0; j < y_num; j++) {
      occupancy[i][j] = 0;
    }
  }
}

void OccupancyPlot::setOccupancy(double x, double y, OccupancyType type){
  // IDの算出
  int i = round( ( x - x_min ) / x_stp);
  int j = round( ( y - y_min ) / y_stp);

  // map上の対応するIDに値を代入
  if (0 <= i && i < x_num && 0 <= j && j < y_num) {
    switch (type) {
    case OccupancyType::NONE:
    case OccupancyType::EMPTY:
      occupancy[i][j] = 0;
      break;
    case OccupancyType::OBSTACLE:
      occupancy[i][j] = 1;
      break;
    case OccupancyType::OPEN:
      occupancy[i][j] = 2;
      break;
    case OccupancyType::CLOSED:
    case OccupancyType::GOAL:
      occupancy[i][j] = 3;
      break;
    }
  }
}

void OccupancyPlot::setOccupancy(vec2_t pos, OccupancyType type){
  setOccupancy(pos.x(), pos.y(), type);
}

void OccupancyPlot::plot(){
  // mapをグラフ上の対応する点に変換
  FILE *fp = fopen("dat/data.dat", "w");
  for (int i = 0; i < x_num; i++) {
    for (int j = 0; j < y_num; j++) {
      fprintf(fp, "%d ", occupancy[i][y_num - j - 1]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);

  // 描画
  fprintf(p.gp, "plot \"dat/data.dat\" matrix w image notitle,\\\n");
  int count = 0;
  for(int i = 0; i <= x_num; i++) {
    std::string file_name = "dat/tmp" + std::to_string(count) + ".dat";
    FILE       *fp        = fopen(file_name.c_str(), "w");

    vec2_t point[2];
    point[0] = cartesianToGraph(x_min - x_stp / 2.0 + x_stp * i, y_min - y_stp / 2.0);
    point[1] = cartesianToGraph(x_min - x_stp / 2.0 + x_stp * i, y_max + y_stp / 2.0);
    fprintf(fp, "%f %f\n", point[0].x(), point[0].y() );
    fprintf(fp, "%f %f\n", point[1].x(), point[1].y() );
    fclose(fp);
    fprintf(p.gp, "\"%s\" with lines lt -1 lw 2 lc \"white\" notitle,\\\n", file_name.c_str() );

    count++;
  }
  for(int i = 0; i <= y_num; i++) {
    std::string file_name = "dat/tmp" + std::to_string(count) + ".dat";
    FILE       *fp        = fopen(file_name.c_str(), "w");

    vec2_t point[2];
    point[0] = cartesianToGraph(x_min - x_stp / 2.0, y_min - y_stp / 2.0 + y_stp * i);
    point[1] = cartesianToGraph(x_max + x_stp / 2.0, y_min - y_stp / 2.0 + y_stp * i);
    fprintf(fp, "%f %f\n", point[0].x(), point[0].y() );
    fprintf(fp, "%f %f\n", point[1].x(), point[1].y() );
    fclose(fp);
    if(i != y_num) {
      fprintf(p.gp, "\"%s\" with lines lt -1 lw 2 lc \"white\" notitle,\\\n", file_name.c_str() );
    }else{
      fprintf(p.gp, "\"%s\" with lines lt -1 lw 2 lc \"white\" notitle\n", file_name.c_str() );
    }

    count++;
  }
  fflush(p.gp);
}

vec2_t OccupancyPlot::cartesianToGraph(vec2_t point){
  vec2_t p;
  double x = ( y_max - point.y() ) / y_stp;
  double y = ( point.x() - x_min ) / x_stp;
  p << x, y;
  return p;
}

vec2_t OccupancyPlot::cartesianToGraph(double x, double y){
  vec2_t p;
  p << x, y;
  return cartesianToGraph(p);
}

}