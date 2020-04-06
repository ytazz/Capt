#include "step_plot.h"

using namespace std;

namespace Capt {

StepPlot::StepPlot(Model *model, Param *param, Grid *grid)
  : model(model), param(param), grid(grid) {
  p("unset key");
  p("unset border");
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
  fprintf(p.gp, "set xrange [0:%d]\n", x_num);
  fprintf(p.gp, "set yrange [0:%d]\n", y_num);

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
  // p("set grid front mxtics mytics lw 2 lt -1 lc rgb 'white'");
  // p("set grid front mxtics mytics lw 2 lt -1 lc rgb 'gray90'");

  // xy軸の目盛りの値を再設定
  std::string x_tics, y_tics;
  for (int i = 0; i <= x_num; i++) {
    x_tics += "\"\" " + str(i);
    if (i != x_num)
      x_tics += ", ";
  }
  for (int i = 0; i <= y_num; i++) {
    y_tics += "\"\" " + str(i);
    if (i != y_num)
      y_tics += ", ";
  }
  fprintf(p.gp, "set xtics add (%s)\n", x_tics.c_str() );
  fprintf(p.gp, "set ytics add (%s)\n", y_tics.c_str() );
  fprintf(p.gp, "set ytics add (\"0\" 10)\n");
  fprintf(p.gp, "set ytics add (\"0.5\" 20)\n");
  fprintf(p.gp, "set ytics add (\"1.0\" 30)\n");
  fprintf(p.gp, "set xtics add (\"0\" 20)\n");
  fprintf(p.gp, "set xtics add (\"-0.5\" 30)\n");
  fprintf(p.gp, "set xtics add (\"0.5\" 10)\n");

  // 変数の初期化
  footstep_r.clear();
  footstep_l.clear();
  icp.clear();
  cop.clear();
}

StepPlot::~StepPlot() {
}

std::string StepPlot::str(float val){
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

  // 描画ファイル(.dat)への書き込み
  std::string name = "";
  FILE       *fp   = NULL;
  for(int i = 0; i < x_num; i++) {
    name = "dat/axis_x_" + std::to_string(i) + ".dat";
    fp   = fopen(name.c_str(), "w");

    vec2_t point[2];
    point[0] = cartesianToGraph(x_min + x_stp / 2 + x_stp * i, y_min - y_stp / 2);
    point[1] = cartesianToGraph(x_min + x_stp / 2 + x_stp * i, y_max - y_stp / 2);
    fprintf(fp, "%f %f\n", point[0].x(), point[0].y() );
    fprintf(fp, "%f %f\n", point[1].x(), point[1].y() );
    fclose(fp);
  }
  for(int i = 0; i < y_num; i++) {
    name = "dat/axis_y_" + std::to_string(i) + ".dat";
    fp   = fopen(name.c_str(), "w");

    vec2_t point[2];
    point[0] = cartesianToGraph(x_min + x_stp / 2, y_min - y_stp / 2 + y_stp * i);
    point[1] = cartesianToGraph(x_max + x_stp / 2, y_min - y_stp / 2 + y_stp * i);
    fprintf(fp, "%f %f\n", point[0].x(), point[0].y() );
    fprintf(fp, "%f %f\n", point[1].x(), point[1].y() );
    fclose(fp);
  }
  for(size_t i = 0; i < footstep_r.size(); i++) {
    name = "dat/footstep_r_" + std::to_string(i) + ".dat";
    fp   = fopen(name.c_str(), "w");
    for(size_t j = 0; j < foot_r.size(); j++) {
      vec2_t point = cartesianToGraph(footstep_r[i] + foot_r[j]);
      fprintf(fp, "%f %f\n", point.x(), point.y() );
    }
    fclose(fp);
  }
  for(size_t i = 0; i < footstep_l.size(); i++) {
    name = "dat/footstep_l_" + std::to_string(i) + ".dat";
    fp   = fopen(name.c_str(), "w");
    for(size_t j = 0; j < foot_l.size(); j++) {
      vec2_t point = cartesianToGraph(footstep_l[i] + foot_l[j]);
      fprintf(fp, "%f %f\n", point.x(), point.y() );
    }
    fclose(fp);
  }
  fp = fopen("dat/icp.dat", "w");
  for(size_t i = 0; i < icp.size(); i++) {
    vec2_t point = cartesianToGraph(icp[i]);
    fprintf(fp, "%f %f\n", point.x(), point.y() );
  }
  fclose(fp);
  fp = fopen("dat/cop.dat", "w");
  for(size_t i = 0; i < cop.size(); i++) {
    vec2_t point = cartesianToGraph(cop[i]);
    fprintf(fp, "%f %f\n", point.x(), point.y() );
  }
  fclose(fp);

  // 描画
  fprintf(p.gp, "plot ");
  for(int i = 0; i < x_num; i++) {
    fprintf(p.gp, "\"dat/axis_x_%d.dat\" with lines lt -1 lw 2 lc \"gray90\" notitle,\\\n", (int)i);
  }
  for(int i = 0; i < y_num; i++) {
    fprintf(p.gp, "\"dat/axis_y_%d.dat\" with lines lt -1 lw 2 lc \"gray90\" notitle,\\\n", (int)i);
  }
  for(size_t i = 0; i < footstep_r.size(); i++) {
    if(i == 0) {
      fprintf(p.gp, "     \"dat/footstep_r_%d.dat\" with lines lw 2 lc \"green\" title \"right foot\",\\\n", (int)i);
    }else{
      fprintf(p.gp, "     \"dat/footstep_r_%d.dat\" with lines lw 2 lc \"green\" notitle,\\\n", (int)i);
    }
  }
  for(size_t i = 0; i < footstep_l.size(); i++) {
    if(i == 0) {
      fprintf(p.gp, "     \"dat/footstep_l_%d.dat\" with lines lw 2 lc \"red\" title \"left foot\",\\\n", (int)i);
    }else{
      fprintf(p.gp, "     \"dat/footstep_l_%d.dat\" with lines lw 2 lc \"red\" notitle,\\\n", (int)i);
    }
  }
  fprintf(p.gp, "     \"dat/icp.dat\"         with lines  lw 2 lc \"dark-blue\" title \"ICP\",\\\n");
  fprintf(p.gp, "     \"dat/cop.dat\"         with points ps 1 lc \"red\"       title \"CoP\"\n");
  fflush(p.gp);
}

vec2_t StepPlot::cartesianToGraph(vec2_t point){
  return vec2_t(( y_max - point.y() ) / y_stp, ( point.x() - x_min ) / x_stp);
}

vec2_t StepPlot::cartesianToGraph(float x, float y){
  return cartesianToGraph(vec2_t(x, y));
}

}