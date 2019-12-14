#include "step_plot.h"

using namespace std;

namespace Capt {

StepPlot::StepPlot(Model *model, Param *param, Grid *grid)
  : model(model), param(param), grid(grid) {
  p("unset key");
  p("set encoding utf8");

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
  fprintf(p.gp, "set xrange [0:%d]\n", x_num - 1);
  fprintf(p.gp, "set yrange [0:%d]\n", y_num - 1);

  // 軸ラベル設定
  p("set xlabel 'y [m]'");
  p("set ylabel 'x [m]'");
  p("set xlabel font \"Arial,10\"");
  p("set ylabel font \"Arial,10\"");
  p("set tics   font \"Arial,10\"");

  // 座標軸の目盛り設定
  p("set xtics 1");
  p("set ytics 1");
  p("set mxtics 2");
  p("set mytics 2");
  p("set xtics scale 0,0.001");
  p("set ytics scale 0,0.001");
  // p("set grid front mxtics mytics lw 2 lt -1 lc rgb 'white'");
  // p("set grid front mxtics mytics lw 2 lt -1 lc rgb 'gray90'");

  // xy軸の目盛りの値を再設定
  std::string x_tics, y_tics;
  for (int i = 0; i < x_num; i++) {
    x_tics += "\"\" " + str(i);
    if (i != x_num - 1)
      x_tics += ", ";
  }
  for (int i = 0; i < y_num; i++) {
    y_tics += "\"\" " + str(i);
    if (i != y_num - 1)
      y_tics += ", ";
  }
  fprintf(p.gp, "set xtics add (%s)\n", x_tics.c_str() );
  fprintf(p.gp, "set ytics add (%s)\n", y_tics.c_str() );
  fprintf(p.gp, "set xtics add (\"%1.1lf\" 0, \"%1.1lf\" %d)\n", y_max, y_min, ( y_num - 1 ) );
  fprintf(p.gp, "set ytics add (\"%1.1lf\" 0, \"%1.1lf\" %d)\n", x_min, x_max, ( x_num - 1 ) );

  // 変数の初期化
  footstep_r.clear();
  footstep_l.clear();
  icp.clear();
  cop.clear();
}

StepPlot::~StepPlot() {
}

std::string StepPlot::str(double val){
  return std::to_string(val);
}

std::string StepPlot::str(int val){
  return std::to_string(val);
}

void StepPlot::setOutput(std::string type) {
  if (type == "gif") {
    p("set terminal gif animate optimize delay 100 size 600,900");
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

void StepPlot::setFootR(vec2_t foot_r){
  this->footstep_r.push_back(foot_r);
}

void StepPlot::setFootR(arr2_t foot_r){
  for(size_t i = 0; i < foot_r.size(); i++) {
    setFootR(foot_r[i]);
  }
}

void StepPlot::setFootL(vec2_t foot_l){
  this->footstep_l.push_back(foot_l);
}

void StepPlot::setFootL(arr2_t foot_l){
  for(size_t i = 0; i < foot_l.size(); i++) {
    setFootL(foot_l[i]);
  }
}

void StepPlot::setIcp(vec2_t icp){
  this->icp.push_back(icp);
}

void StepPlot::setIcp(arr2_t icp){
  for(size_t i = 0; i < icp.size(); i++) {
    setIcp(icp[i]);
  }
}

void StepPlot::setCop(vec2_t cop){
  this->cop.push_back(cop);
}

void StepPlot::setCop(arr2_t cop){
  for(size_t i = 0; i < cop.size(); i++) {
    setCop(cop[i]);
  }
}

void StepPlot::setSequence(std::vector<Sequence> seq){
  for(size_t i = 0; i < seq.size(); i++) {
    if(seq[i].suf == Foot::FOOT_R) {
      setFootR(vec3Tovec2(seq[i].pos) );
    }else{
      setFootL(vec3Tovec2(seq[i].pos) );
    }
    setIcp(vec3Tovec2(seq[i].icp) );
    setCop(vec3Tovec2(seq[i].cop) );
  }
}

void StepPlot::plot(){
  arr2_t foot_r;
  arr2_t foot_l;
  model->read(&foot_r, "foot_r");
  model->read(&foot_l, "foot_l");

  // 描画対象の追加
  fprintf(p.gp, "plot ");
  for(size_t i = 0; i < footstep_r.size(); i++) {
    fprintf(p.gp, "'-' with lines lw 2 lc \"green\",");
  }
  for(size_t i = 0; i < footstep_l.size(); i++) {
    fprintf(p.gp, "'-' with lines linewidth 2 lc \"red\",");
  }
  fprintf(p.gp, "'-' with lines  lw 2 lc \"dark-blue\",");
  fprintf(p.gp, "'-' with points ps 1 lc \"red\"\n");

  // 描画
  // footstep_r
  for(size_t i = 0; i < footstep_r.size(); i++) {
    for(size_t j = 0; j < foot_r.size(); j++) {
      vec2_t point = cartesianToGraph(footstep_r[i] + foot_r[j]);
      fprintf(p.gp, "%f %f\n", point.x(), point.y() );
    }
    fprintf(p.gp, "e\n");
  }
  // footstep_l
  for(size_t i = 0; i < footstep_l.size(); i++) {
    for(size_t j = 0; j < foot_l.size(); j++) {
      vec2_t point = cartesianToGraph(footstep_l[i] + foot_l[j]);
      fprintf(p.gp, "%f %f\n", point.x(), point.y() );
    }
    fprintf(p.gp, "e\n");
  }
  // icp
  for(size_t i = 0; i < icp.size(); i++) {
    vec2_t point = cartesianToGraph(icp[i]);
    fprintf(p.gp, "%f %f\n", point.x(), point.y() );
  }
  fprintf(p.gp, "e\n");
  // cop
  for(size_t i = 0; i < cop.size(); i++) {
    vec2_t point = cartesianToGraph(cop[i]);
    fprintf(p.gp, "%f %f\n", point.x(), point.y() );
  }
  fprintf(p.gp, "e\n");

  fflush(p.gp);
}

vec2_t StepPlot::cartesianToGraph(vec2_t point){
  vec2_t p;
  double x = ( y_max - point.y() ) / y_stp;
  double y = ( point.x() - x_min ) / x_stp;
  p << x, y;
  return p;
}

vec2_t StepPlot::cartesianToGraph(double x, double y){
  vec2_t p;
  p << x, y;
  return cartesianToGraph(p);
}

}