#include "cr_plot.h"

using namespace std;

namespace Capt {

CRPlot::CRPlot(Model *model, Param *param)
  : model(model), param(param){
  // ファイル形式の確認
  // p("unset key");
  p("set encoding utf8");
  p("set terminal qt size 600,600");

  param->read(&swf_x_min, "swf_x_min");
  param->read(&swf_x_max, "swf_x_max");
  param->read(&swf_y_min, "swf_y_min");
  param->read(&swf_y_max, "swf_y_max");

  param->read(&cop_x_min, "cop_x_min");
  param->read(&cop_x_max, "cop_x_max");
  param->read(&cop_y_min, "cop_y_min");
  param->read(&cop_y_max, "cop_y_max");

  param->read(&exc_x_min, "exc_x_min");
  param->read(&exc_x_max, "exc_x_max");
  param->read(&exc_y_min, "exc_y_min");
  param->read(&exc_y_max, "exc_y_max");

  param->read(&grid_x_min, "grid_x_min");
  param->read(&grid_x_max, "grid_x_max");
  param->read(&grid_y_min, "grid_y_min");
  param->read(&grid_y_max, "grid_y_max");

  c_num = 5;

  // グラフサイズ設定
  p("set size square");

  // 軸ラベル設定
  p("set xlabel 'y [m]'");
  p("set ylabel 'x [m]'");
  p("set xlabel font \"Arial,15\"");
  p("set ylabel font \"Arial,15\"");
  p("set tics   font \"Arial,15\"");
  p("set cblabel font \"Arial,15\"");
  p("set key    font \"Arial,15\"");

  // 座標軸の目盛り設定
  p("set xtics 1");
  p("set ytics 1");
  p("set mxtics 2");
  p("set mytics 2");
  p("set xtics scale 0,0.001");
  p("set ytics scale 0,0.001");
  fprintf(p.gp, "set xrange [%f:%f]\n", -grid_y_max, -grid_y_min);
  fprintf(p.gp, "set yrange [%f:%f]\n",  grid_x_min,  grid_x_max);

  // カラーバーの設定
  // p("set palette gray negative");
  fprintf(p.gp, "set palette defined ( 0 '#ffffff', 1 '#cbfeff', 2 '#68fefe', 3 '#0097ff', 4 '#0000ff')\n");
  fprintf(p.gp, "set cbrange[0:%d]\n", c_num);
  fprintf(p.gp, "set cbtics 0.5\n");
  fprintf(p.gp, "set palette maxcolors %d\n", c_num);
  fprintf(p.gp, "set cbtics scale 0,0.001\n");
  //fprintf(p.gp, "set cblabel \"N-step capture point\"\n");

  setFootRegion();
}

CRPlot::~CRPlot() {
}

std::string CRPlot::str(float val){
  return std::to_string(val);
}

std::string CRPlot::str(int val){
  return std::to_string(val);
}

void CRPlot::setOutput(std::string type) {
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

void CRPlot::setFootRegion(){
  vec2_t vertex[9];
  vertex[0] << swf_x_min, swf_y_min;
  vertex[1] << exc_x_min, swf_y_min;
  vertex[2] << exc_x_min, exc_y_max;
  vertex[3] << exc_x_max, exc_y_max;
  vertex[4] << exc_x_max, swf_y_min;
  vertex[5] << swf_x_max, swf_y_min;
  vertex[6] << swf_x_max, swf_y_max;
  vertex[7] << swf_x_min, swf_y_max;
  vertex[8] << swf_x_min, swf_y_min;
  for(int i = 0; i < 9; i++)
    vertex[i] = cartesianToGraph(vertex[i]);

  FILE *fp = fopen("dat/foot_region.dat", "w");
  for(int i = 0; i < 9; i++) {
    fprintf(fp, "%lf %lf\n", vertex[i].x(), vertex[i].y() );
  }
  fclose(fp);
}

void CRPlot::setState(State state){
  setIcp(state.icp);
  setSwf(vec3Tovec2(state.swf) );
}

void CRPlot::setIcp(vec2_t icp){
  vec2_t point = cartesianToGraph(icp);
  FILE  *fp    = fopen("dat/icp.dat", "w");
  fprintf(fp, "%lf %lf\n", point.x(), point.y() );
  fclose(fp);
}

void CRPlot::setSwf(vec2_t swf){
  arr2_t foot_r, foot_l;
  model->read(&foot_r, "foot_r");
  model->read(&foot_l, "foot_l", swf);

  FILE *fp;
  fp = fopen("dat/foot_r.dat", "w");
  vec2_t point;
  for (size_t i = 0; i < foot_r.size(); i++) {
    // グラフ座標に合わせる
    point = cartesianToGraph(foot_r[i]);
    fprintf(fp, "%lf %lf\n", point.x(), point.y() );
  }
  fclose(fp);
  fp = fopen("dat/foot_l.dat", "w");
  for (size_t i = 0; i < foot_l.size(); i++) {
    // グラフ座標に合わせる
    point = cartesianToGraph(foot_l[i]);
    fprintf(fp, "%lf %lf\n", point.x(), point.y() );
  }
  fclose(fp);
}

void CRPlot::setCaptureInput(Input in, int nstep){
  cap_input.push_back(make_pair(in, nstep));
}

void CRPlot::plot(){
  // mapをグラフ上の対応する点に変換
  FILE *fp;
  fp = fopen("dat/data.dat", "w");
  for(int i = 0; i < (int)cap_input.size(); i++){
    vec2_t cop = cartesianToGraph(cap_input[i].first.cop.x(), cap_input[i].first.cop.y());
    vec2_t swf = cartesianToGraph(cap_input[i].first.swf.x(), cap_input[i].first.swf.y());
    fprintf(fp, "%f %f %f %f %d\n", cop.x(), cop.y(), swf.x(), swf.y(), cap_input[i].second);
  }
  fclose(fp);

  // 描画
  fprintf(p.gp, "plot \"dat/data.dat\" using ($5 == 0 ? 1/0 : $1):($5 == 0 ? 1/0 : $2):($5) with points palette pt 5 ps 0.5 notitle,\\\n");
  fprintf(p.gp, "     \"dat/data.dat\" using ($5 == 0 ? 1/0 : $3):($5 == 0 ? 1/0 : $4):($5) with points palette pt 5 ps 0.5 notitle,\\\n");
  fprintf(p.gp, "     \"dat/foot_region.dat\" with lines  lw 1 lc \"dark-blue\" notitle,\\\n");
  fprintf(p.gp, "     \"dat/foot_r.dat\"      with lines  lw 1 lc \"black\"     notitle,\\\n");
  fprintf(p.gp, "     \"dat/foot_l.dat\"      with lines  lt 0 dt 1 lw 2 lc \"black\" notitle,\\\n");
  fprintf(p.gp, "     \"dat/icp.dat\"         with points pt 1 lc 1 ps 2        notitle\n");
  fflush(p.gp);
}

vec2_t CRPlot::cartesianToGraph(vec2_t point){
  return vec2_t(-point.y(), point.x());
}

vec2_t CRPlot::cartesianToGraph(float x, float y){
  return cartesianToGraph(vec2_t(x, y));
}

}
