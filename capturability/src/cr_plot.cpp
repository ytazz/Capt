#include "cr_plot.h"

using namespace std;

namespace Capt {

CRPlot::CRPlot(Model model, Param param)
  : model(model), param(param) {
  p("set title \"Capture Region (polar coordinate)\"");
  // p("unset key");
  // p("set encoding utf8");
  // p("set size square");

  x_min  = param.getVal("icp_x", "min");
  x_max  = param.getVal("icp_x", "max");
  x_step = param.getVal("icp_x", "step");
  y_min  = param.getVal("icp_y", "min");
  y_max  = param.getVal("icp_y", "max");
  y_step = param.getVal("icp_y", "step");
  if (strcmp(param.getStr("coordinate", "type").c_str(), "cartesian") == 0) {

    std::string xrange = "[" + str(x_min - x_step / 2.0) + ":" + str(x_max + x_step / 2.0) + "]";
    std::string yrange = "[" + str(y_min - y_step / 2.0) + ":" + str(y_max + y_step / 2.0) + "]";

    // p("set xtics "  + str(x_min) + ", " + str(x_step * 2.0) );
    // p("set ytics " + str(y_step * 2.0) );
    // p("set x2tics " + str(x_min - x_step / 2.0) + ", " + str(x_step) );
    // printf("x_min %s\n", str(x_min).c_str() );
    // printf("x_max %s\n", str(x_max).c_str() );
    // printf("x_step %s\n", str(x_step).c_str() );
    // p("set format x2 \"\"");
    // p("set xrange " + xrange);
    // p("set yrange " + yrange);
    // p("set x2tics scale 0");
    //
    // p("set grid x2tics");

    // p("set terminal svg");
    // p("set output 'plot.svg'");
    p("set size square");
    p("set palette gray negative");
    p("set autoscale xfix");
    p("set autoscale yfix");
    p("set xtics 1");
    p("set ytics 1");
    p("set title \"Resolution Matrix for E\"");

    p("set cbrange[0:4]");
    p("set cbtics 1");

    p("set tics scale 0,0.1");
    p("set mxtics 2");
    p("set mytics 2");
    p("set grid front mxtics mytics lw 2 lt -1 lc rgb 'white'");

    // rename
    p("set xtics add (\"1\" 0, \"\" 1, \"\" 2, \"\" 3, \"5\" 4, \"\" 5, \"\" 6, \"\" 7, \"\" 8, \"10\" 9, \"\" 10,)");

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

  // com.setPolar(0.04, 30 * 3.14159 / 180);
  com.setPolar(0.01, 30 * 3.14159 / 180);
}

CRPlot::~CRPlot() {
}

std::string CRPlot::str(double val){
  return std::to_string(val);
}

void CRPlot::setOutput(std::string type) {
  if (type == "gif") {
    p("set terminal gif animate optimize delay 30 size 600,900");
    p("set output 'plot.gif'");
  }
  if (type == "svg") {
    // p("set terminal svg");
    // p("set output 'plot.svg'");
    fprintf(p.gp, "plot \"datafile.dat\" matrix w image notitle\n");
    // fflush(p.gp);
  }
  if (type == "eps") {
    p("set terminal postscript eps enhanced");
    p("set output 'plot.eps'");
  }
}

void CRPlot::plot(){
  FILE  *fp = fopen("data.dat", "w");
  double x  = x_min, y = y_min;
  while(x < x_max + x_step / 2.0) {
    y = y_min;
    while(y < y_max + y_step / 2.0) {
      fprintf(fp, "%d ", rand() % 5);
      y += y_step;
    }
    fprintf(fp, "\n");
    x += x_step;
  }
  fclose(fp);
  fprintf(p.gp, "plot \"data.dat\" matrix w image notitle\n");
}
}