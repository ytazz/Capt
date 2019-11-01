#include "cr_plot.h"

using namespace std;

namespace Capt {

CRPlot::CRPlot(Model model, Param param)
  : model(model), param(param), grid(param) {
  // ファイル形式の確認
  if (param.getStr("coordinate", "type") == "polar") {
    printf("Error: coordinate type \"polar\" is not supported.\n");
  }

  p("unset key");
  p("set encoding utf8");

  x_min  = param.getVal("swf_x", "min");
  x_max  = param.getVal("swf_x", "max");
  x_step = param.getVal("swf_x", "step");
  x_num  = param.getVal("swf_x", "num");
  y_min  = param.getVal("swf_y", "min");
  y_max  = param.getVal("swf_y", "max");
  y_step = param.getVal("swf_y", "step");
  y_num  = param.getVal("swf_y", "num");
  c_num  = 5;

  // グラフサイズ設定
  p("set size square");
  fprintf(p.gp, "set xrange [0:%d]\n", x_num - 1);
  fprintf(p.gp, "set yrange [0:%d]\n", y_num - 1);

  // 軸ラベル設定
  p("set xlabel 'y [m]'");
  p("set ylabel 'x [m]'");
  p("set xlabel font \"Arial,15\"");
  p("set ylabel font \"Arial,15\"");
  p("set tics   font \"Arial,15\"");

  // 座標軸の目盛り設定
  p("set xtics 1");
  p("set ytics 1");
  p("set mxtics 2");
  p("set mytics 2");
  p("set xtics scale 0,0.001");
  p("set ytics scale 0,0.001");
  // p("set grid front mxtics mytics lw 2 lt -1 lc rgb 'white'");
  p("set grid front mxtics mytics lw 2 lt -1 lc rgb 'gray90'");

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
}

CRPlot::~CRPlot() {
}

std::string CRPlot::str(double val){
  return std::to_string(val);
}

std::string CRPlot::str(int val){
  return std::to_string(val);
}

void CRPlot::setOutput(std::string type) {
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

void CRPlot::setFootR(vec2_t foot_r){
  vec2_t foot_r_ = cartesianToGraph(foot_r);
  footstep_r.push_back(foot_r_);
}

void CRPlot::setFootL(vec2_t foot_l){
  vec2_t foot_l_ = cartesianToGraph(foot_l);
  footstep_l.push_back(foot_l_);
}

void CRPlot::setIcp(vec2_t icp){
  vec2_t icp_ = cartesianToGraph(icp);
  icp.push_back(icp_);
}

void CRPlot::plot(){
  // 描画対象の追加
  fprintf(p.gp, "plot \"dat/icp.dat\" with lines  lw 2 lc \"dark-blue\" title \"icp\",\\\n");
  fprintf(p.gp, "     \"dat/foot_r.dat\"      with lines  lw 2 lc \"black\"     title \"foot_su\",\\\n");
  fprintf(p.gp, "     \"dat/foot_l.dat\"      with lines  lw 2 lc \"black\"     title \"foot_sw\",\\\n");
  fprintf(p.gp, "     \"dat/icp.dat\"         with points pt 2 lc 1             title \"icp\"\n");

  // 描画
  fflush(p.gp);
}

vec2_t CRPlot::cartesianToGraph(vec2_t point){
  vec2_t p;
  double x = -point.y / y_step + ( y_num - 1 ) / 2;
  double y = +point.x / x_step + ( x_num - 1 ) / 2;
  p.setCartesian(x, y);
  return p;
}

vec2_t CRPlot::cartesianToGraph(double x, double y){
  vec2_t p;
  p.setCartesian(x, y);
  return cartesianToGraph(p);
}

}