#include "cr_plot.h"

using namespace std;

namespace Capt {

CRPlot::CRPlot(Model *model, Param *param, Grid *grid)
  : model(model), param(param), grid(grid) {
  // ファイル形式の確認
  p("unset key");
  p("set encoding utf8");

  param->read(&x_min, "icp_x_min");
  param->read(&x_max, "icp_x_max");
  param->read(&x_stp, "icp_x_stp");
  param->read(&x_num, "icp_x_num");
  param->read(&y_min, "icp_y_min");
  param->read(&y_max, "icp_y_max");
  param->read(&y_stp, "icp_y_stp");
  param->read(&y_num, "icp_y_num");

  c_num = 5;

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
  p("set grid front mxtics mytics lw 2 lt -1 lc rgb 'white'");
  // p("set grid front mxtics mytics lw 2 lt -1 lc rgb 'gray80'");

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

  // カラーバーの設定
  // p("set palette gray negative");
  fprintf(p.gp, "set palette defined ( 0 '#ffffff', 1 '#cbfeff', 2 '#68fefe', 3 '#0097ff', 4 '#3666fe')\n");
  fprintf(p.gp, "set cbrange[0:%d]\n", c_num);
  fprintf(p.gp, "set cbtics 0.5\n");
  fprintf(p.gp, "set palette maxcolors %d\n", c_num);
  fprintf(p.gp, "set cbtics scale 0,0.001\n");

  // カラーバーの目盛りの値を再設定
  std::string c_tics;
  for (int i = 0; i <= c_num; i++) {
    if (i != c_num) {
      c_tics += "\"\" " + str(i);
      c_tics += ", ";
      if(i == 0) {
        c_tics += "\"NONE\" " + str(i + 0.5);
      }else{
        c_tics += "\"" + str(i) + "\" " + str(i + 0.5);
      }
      c_tics += ", ";
    }else{
      c_tics += "\"\" " + str(i);
    }
  }
  fprintf(p.gp, "set cbtics add (%s)\n", c_tics.c_str() );

  initCaptureMap();
  setFootRegion();
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
    p("set terminal gif animate optimize delay 300 size 600,900");
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
  double swf_x_min;
  double swf_x_max;
  double swf_y_min;
  double swf_y_max;
  param->read(&swf_x_min, "swf_x_min");
  param->read(&swf_x_max, "swf_x_max");
  param->read(&swf_y_min, "swf_y_min");
  param->read(&swf_y_max, "swf_y_max");
  swf_x_min -= x_stp / 2.0;
  swf_x_max += x_stp / 2.0;
  swf_y_min -= y_stp / 2.0;
  swf_y_max += y_stp / 2.0;

  vec2_t vertex[5];
  vertex[0] << swf_x_min, swf_y_min;
  vertex[1] << swf_x_max, swf_y_min;
  vertex[2] << swf_x_max, swf_y_max;
  vertex[3] << swf_x_min, swf_y_max;
  vertex[4] << swf_x_min, swf_y_min;
  vertex[0] = cartesianToGraph(vertex[0]);
  vertex[1] = cartesianToGraph(vertex[1]);
  vertex[2] = cartesianToGraph(vertex[2]);
  vertex[3] = cartesianToGraph(vertex[3]);
  vertex[4] = cartesianToGraph(vertex[4]);

  FILE *fp = fopen("dat/foot_region.dat", "w");
  for(int i = 0; i < 5; i++) {
    fprintf(fp, "%lf %lf\n", vertex[i].x(), vertex[i].y() );
  }
  fclose(fp);
}

void CRPlot::setFoot(vec2_t swf){
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

void CRPlot::setIcp(vec2_t icp){
  vec2_t point = cartesianToGraph(icp);
  FILE  *fp    = fopen("dat/icp.dat", "w");
  fprintf(fp, "%lf %lf\n", point.x(), point.y() );
  fclose(fp);
}

void CRPlot::initCaptureMap(){
  capture_map.clear();
  capture_map.resize(x_num);
  for (int i = 0; i < x_num; i++) {
    capture_map[i].resize(y_num);
    for (int j = 0; j < y_num; j++) {
      capture_map[i][j] = 0;
    }
  }
}

void CRPlot::setCaptureMap(double x, double y, int n_step){
  // IDの算出
  int i = ( x - x_min ) / x_stp;
  int j = ( y - y_min ) / y_stp;

  // doubleからintに丸める時の四捨五入
  if ( ( x - x_min ) / x_stp - i >= 0.5)
    i++;
  if ( ( y - y_min ) / y_stp - j >= 0.5)
    j++;

  // map上の対応するIDに値を代入
  if (0 <= i && i < x_num && 0 <= j && j < y_num)
    capture_map[i][j] = n_step;
}

void CRPlot::plot(){
  // mapをグラフ上の対応する点に変換
  FILE *fp = fopen("dat/data.dat", "w");
  for (int i = 0; i < x_num; i++) {
    for (int j = 0; j < y_num; j++) {
      fprintf(fp, "%d ", capture_map[i][y_num - j - 1]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);

  // 描画
  fprintf(p.gp, "plot \"dat/data.dat\" matrix w image notitle,\\\n");
  fprintf(p.gp, "     \"dat/foot_region.dat\" with lines  lw 2 lc \"dark-blue\" title \"foot_region\",\\\n");
  fprintf(p.gp, "     \"dat/foot_r.dat\"      with lines  lw 2 lc \"black\"     title \"foot_su\",\\\n");
  fprintf(p.gp, "     \"dat/foot_l.dat\"      with lines  lw 2 lc \"black\"     title \"foot_sw\",\\\n");
  fprintf(p.gp, "     \"dat/icp.dat\"         with points pt 2 lc 1             title \"icp\"\n");
  fflush(p.gp);
}

vec2_t CRPlot::cartesianToGraph(vec2_t point){
  vec2_t p;
  double x = -point.y() / y_stp + ( y_num - 1 ) / 2;
  double y = +point.x() / x_stp + ( x_num - 1 ) / 2;
  p << x, y;
  return p;
}

vec2_t CRPlot::cartesianToGraph(double x, double y){
  vec2_t p;
  p << x, y;
  return cartesianToGraph(p);
}

}