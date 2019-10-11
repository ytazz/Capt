#include "cr_plot.h"

using namespace std;

namespace Capt {

CRPlot::CRPlot(Model model, Param param)
  : model(model), param(param) {
  p("set title \"Capture Region (polar coordinate)\"");
  // p("unset key");
  // p("set encoding utf8");
  // p("set size square");

  if (strcmp(param.getStr("coordinate", "type").c_str(), "cartesian") == 0) {
    double x_min  = param.getVal("icp_x", "min");
    double x_max  = param.getVal("icp_x", "max");
    double x_step = param.getVal("icp_x", "step");
    double y_min  = param.getVal("icp_y", "min");
    double y_max  = param.getVal("icp_y", "max");
    double y_step = param.getVal("icp_y", "step");

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
    p("set size ratio 1");
    p("set palette gray negative");
    p("set autoscale xfix");
    p("set autoscale yfix");
    p("set xtics 1");
    p("set ytics 1");
    p("set title \"Resolution Matrix for E\"");

    p("set cbrange[0:3]");
    p("set cbtics 1");

    p("set tics scale 0,0.1");
    p("set mxtics 2");
    p("set mytics 2");
    p("set grid front mxtics mytics lw 1.5 lt -1 lc rgb 'white'");

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

void CRPlot::plotGrid(){
  // fprintf(p.gp, "plot ");
  // fprintf(p.gp, "'-' with lines lw 1 lc \"black\"\n");
  // double x,y;
  // for(int i = 0; i < param.getVal("icp_x", "num"); i++) {
  //   for(int j = 0; j < param.getVal("icp_y", "num"); j++) {
  //     fprintf(p.gp, "%f %f\n", x, y );
  //   }
  // }
  // fprintf(p.gp, "e\n");
}

/*
   void CRPlot::animCaptureRegion(State state) {
   state.icp.th = param.getVal("icp_th", "min");

   while (state.icp.th < param.getVal("icp_th", "max") ) {
    state.icp.th += param.getVal("icp_th", "step");
    plotCaptureRegion(state);
   }
   }

   void CRPlot::plotCaptureRegion(State state) {
   // setting
   fprintf(p.gp, "plot ");
   // steppable region
   fprintf(p.gp, "'-' t 'Valid Stepping Region' with lines linewidth 1 "
          "lc \"black\",");
   // icp
   fprintf(p.gp, "'-' t 'Instantaneous Capture Point' with points pointsize 1 "
          "pointtype 6 lc \"blue\",");
   // com
   fprintf(p.gp, "'-' t 'Center of Mass' with points pointsize 1 "
          "pointtype 26 lc \"black\",");
   // support foot
   fprintf(p.gp, "'-' t 'Current Support Foot' with lines linewidth 2 "
          "lc \"black\",");
   // swing foot
   fprintf(p.gp, "'-' t 'Current Swing Foot' with lines linewidth 2 "
          "lc \"black\",");
   // n-step capture point
   int max_step = 3;
   for(int i = 1; i <= max_step; i++) {
    if(i == max_step) {
      fprintf(p.gp, "'-' t '%d-step Capture Point' with points pointsize 0.5 "
              "pointtype 7 lc rgb hsv2rgb(%lf, 1, 1)\n", i, 0.3 * ( i - 1 ) );
    }else{
      fprintf(p.gp, "'-' t '%d-step Capture Point' with points pointsize 0.5 "
              "pointtype 7 lc rgb hsv2rgb(%lf, 1, 1),", i, 0.3 * ( i - 1 ) );
    }
   }

   // plot
   // steppable region
   double th = param.getVal("swf_th", "min");
   while (th <= param.getVal("swf_th", "max") ) {
    th += 0.001;
    fprintf(p.gp, "%f %f\n", th, param.getVal("swf_r", "min") );
   }
   while (th >= param.getVal("swf_th", "min") ) {
    th -= 0.001;
    fprintf(p.gp, "%f %f\n", th, param.getVal("swf_r", "max") );
   }
   fprintf(p.gp, "%f %f\n", th, param.getVal("swf_r", "min") );
   fprintf(p.gp, "e\n");
   // icp
   fprintf(p.gp, "%f %f\n", state.icp.th, state.icp.r);
   fprintf(p.gp, "e\n");
   // com
   fprintf(p.gp, "%f %f\n", com.th, com.r);
   fprintf(p.gp, "e\n");
   // support foot
   std::vector<Vector2> foot_r = model.getVec("foot", "foot_r");
   for (size_t i = 0; i < foot_r.size(); i++) {
    fprintf(p.gp, "%f %f\n", foot_r[i].th, foot_r[i].r);
   }
   fprintf(p.gp, "e\n");
   // swing foot
   std::vector<Vector2> foot_l = model.getVec("foot", "foot_l");
   for (size_t i = 0; i < foot_l.size(); i++) {
    fprintf(p.gp, "%f %f\n", ( foot_l[i] + state.swf ).th,
            ( foot_l[i] + state.swf ).r);
   }
   fprintf(p.gp, "e\n");
   // capture point
   for(int i = 1; i <= max_step; i++) {
    std::vector<CaptureSet> region;
    region = capturability.getCaptureRegion(state, i);
    if (!region.empty() ) {
      for (size_t j = 0; j < region.size(); j++) {
        fprintf(p.gp, "%f %f\n", region[j].swf.th, region[j].swf.r);
      }
    }
    fprintf(p.gp, "e\n");
   }

   // flush
   fflush(p.gp);
   }

   void CRPlot::plotCaptureIcp(State state) {
   // setting
   fprintf(p.gp, "plot ");
   // support foot
   fprintf(p.gp, "'-' t 'Current Support Foot' with lines linewidth 2 "
          "lc \"black\",");
   // swing foot
   fprintf(p.gp, "'-' t 'Current Swing Foot' with lines linewidth 2 "
          "lc \"black\",");
   // capturable ICP
   fprintf(p.gp, "'-' t 'Capturable ICP' with points pointsize 0.5 "
          "pointtype 7 lc \"blue\"\n");

   // plot
   // support foot
   std::vector<Vector2> foot_r = model.getVec("foot", "foot_r");
   for (size_t i = 0; i < foot_r.size(); i++) {
    fprintf(p.gp, "%f %f\n", foot_r[i].th, foot_r[i].r);
   }
   fprintf(p.gp, "e\n");
   // swing foot
   std::vector<Vector2> foot_l = model.getVec("foot", "foot_l");
   for (size_t i = 0; i < foot_l.size(); i++) {
    fprintf(p.gp, "%f %f\n", ( foot_l[i] + state.swf ).th,
            ( foot_l[i] + state.swf ).r);
   }
   fprintf(p.gp, "e\n");
   // capturable ICP
   float icp_r  = param.getVal("icp_r", "min");
   float icp_th = param.getVal("icp_th", "min");
   while (icp_r < param.getVal("icp_r", "max") ) {
    icp_th = param.getVal("icp_th", "min");
    while (icp_th < param.getVal("icp_th", "max") ) {
      state.icp.setPolar(icp_r, icp_th);
      if (capturability.capturable(state, 0) ) {
        fprintf(p.gp, "%f %f\n", icp_th, icp_r);
      }
      icp_th += param.getVal("icp_th", "step");
    }
    icp_r += param.getVal("icp_r", "step");
   }
   fprintf(p.gp, "e\n");

   // flush
   fflush(p.gp);
   }
 */
}