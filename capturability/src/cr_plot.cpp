#include "cr_plot.h"

using namespace std;

namespace Capt {

CRPlot::CRPlot(Model model, Param param)
  : model(model), param(param) {
  p("set title \"Capture Region\"");
  // p("unset key");
  p("set encoding utf8");

  x_min  = param.getVal("icp_x", "min");
  x_max  = param.getVal("icp_x", "max");
  x_step = param.getVal("icp_x", "step");
  x_num  = param.getVal("icp_x", "num");
  y_min  = param.getVal("icp_y", "min");
  y_max  = param.getVal("icp_y", "max");
  y_step = param.getVal("icp_y", "step");
  y_num  = param.getVal("icp_y", "num");
  initCaptureMap();

  if (strcmp(param.getStr("coordinate", "type").c_str(), "cartesian") == 0) {
    // グラフサイズ設定
    p("set size square");
    p("set autoscale xfix");
    p("set autoscale yfix");

    // 軸ラベル設定
    p("set xlabel 'y [m]'");
    p("set ylabel 'x [m]'");

    // 座標軸の目盛り設定
    p("set xtics 1");
    p("set ytics 1");
    p("set mxtics 2");
    p("set mytics 2");
    p("set tics scale 0,0.001");
    p("set grid front mxtics mytics lw 2 lt -1 lc rgb 'white'");

    // カラーバーの設定
    p("set palette gray negative");
    p("set cbrange[0:4]");
    p("set cbtics 1");

    // 目盛りの値を再設定
    std::string x_tics, y_tics;
    for(int i = 0; i < x_num; i++) {
      x_tics += "\"\" " + str(i);
      if(i != x_num - 1)
        x_tics += ", ";
    }
    for(int i = 0; i < y_num; i++) {
      y_tics += "\"\" " + str(i);
      if(i != y_num - 1)
        y_tics += ", ";
    }
    fprintf(p.gp, "set xtics add (%s)\n", x_tics.c_str() );
    fprintf(p.gp, "set ytics add (%s)\n", y_tics.c_str() );
    p("set xtics add (\"0.2\" 0, \"0.1\" 5, \"0\" 10, \"-0.1\" 15, \"-0.2\" 20)");
    p("set ytics add (\"-0.2\" 0, \"-0.1\" 5, \"0\" 10, \"0.1\" 15, \"0.2\" 20)");

  }else if (strcmp(param.getStr("coordinate", "type").c_str(), "cartesian") == 0) {
    p("set xtics 0.05");
    p("set ytics 0.05");

    std::string step_r  = str(param.getVal("swf_r", "step") );
    std::string step_th = str(param.getVal("swf_th", "step") );

    p("set polar");
    p("set theta top");
    p("set theta counterclockwise");
    p("set grid polar " + step_th + "lt 1 lc \"gray\"");
    p("set rrange [0:0.20]");
    p("set trange [0:6.28]");
    p("set rtics scale 0");
    p("set rtics " + step_r);
    p("set rtics format \"\"");
    p("unset raxis");
  }

  double g = model.getVal("environment", "gravity");
  double h = model.getVal("physics", "com_height");
  omega = sqrt(g / h);
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
    p("set terminal gif animate optimize delay 30 size 600,900");
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

void CRPlot::setCaptureRegion(){
  initCaptureMap();
  setCaptureMap(0.1, 0.1, 3);
  plot();
}

void CRPlot::initCaptureMap(){
  capture_map.clear();
  capture_map.resize(x_num);
  for(int i = 0; i < x_num; i++) {
    capture_map[i].resize(y_num);
    for(int j = 0; j < y_num; j++) {
      capture_map[i][j] = 0;
    }
  }
}

void CRPlot::setCaptureMap(double x, double y, int n_step){
  // IDの算出
  int i = ( x - x_min ) / x_step;
  int j = ( y - y_min ) / y_step;

  // doubleからintに丸める時の四捨五入
  if( ( x - x_min ) / x_step - i >= 0.5)
    i++;
  if( ( y - y_min ) / y_step - j >= 0.5)
    j++;

  // map上の対応するIDに値を代入
  if(0 <= i && i < x_num && 0 <= j && j < y_num)
    capture_map[i][j] = n_step;
}

void CRPlot::plot(){
  // mapをグラフ上の対応する点に変換
  FILE *fp = fopen("data.dat", "w");
  for(int i = 0; i < x_num; i++) {
    for(int j = 0; j < y_num; j++) {
      fprintf(fp, "%d ", capture_map[i][y_num - j - 1]);
    }
    fprintf(fp, "\n");
  }
  fclose(fp);

  // 描画
  fprintf(p.gp, "plot \"data.dat\" matrix w image notitle\n");
}

}