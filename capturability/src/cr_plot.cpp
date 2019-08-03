#include "cr_plot.h"

using namespace std;

namespace CA {

CRPlot::CRPlot(Model model, Param param)
    : model(model), param(param), capturability(model, param) {
  p("set title \"Capture Region (polar coordinate)\"");
  p("set encoding utf8");
  p("set size square");
  // p("unset key");

  p("set xtics 0.05");
  p("set ytics 0.05");

  p("set polar");
  p("set theta top");
  p("set theta counterclockwise");
  p("set grid polar " + std::to_string(param.getVal("swft_th", "step")) +
    "lt 1 lc \"gray\"");
  p("set rrange [0:0.20]");
  p("set trange [0:6.28]");
  p("set rtics scale 0");
  p("set rtics " + std::to_string(param.getVal("swft_r", "step")));
  p("set rtics format \"\"");
  p("unset raxis");
}

CRPlot::~CRPlot() {}

void CRPlot::setInput(std::string file_name) { capturability.load(file_name); }

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

void CRPlot::plotCaptureRegion() {
  State state;
  state.icp.setPolar(0.06, 0);
  state.swft.setPolar(0.1, 120 * 3.14159 / 180);
  while (state.icp.th < 3.14) {
    // setting
    fprintf(p.gp, "plot ");
    // steppable region
    fprintf(p.gp, "'-' t 'Valid Stepping Region' with lines linewidth 1 "
                  "lc \"black\",");
    // icp
    fprintf(p.gp, "'-' t 'Instantaneous Capture Point' with points pointsize 1 "
                  "pointtype 7 lc \"blue\",");
    // support foot
    fprintf(p.gp, "'-' t 'Current Support Foot' with lines linewidth 2 "
                  "lc \"black\",");
    // swing foot
    fprintf(p.gp, "'-' t 'Current Swing Foot' with lines linewidth 2 "
                  "lc \"black\",");
    // capture region
    int step_num = 0;
    fprintf(p.gp, "'-' t '%d-step Capture Region' with points pointsize 0.5 "
                  "pointtype 7 lc \"%s\"\n",
            step_num, "gray");

    // plot
    // steppable region
    double th = param.getVal("swft_th", "min");
    while (th <= param.getVal("swft_th", "max")) {
      th += 0.001;
      fprintf(p.gp, "%f %f\n", th, param.getVal("swft_r", "min"));
    }
    while (th >= param.getVal("swft_th", "min")) {
      th -= 0.001;
      fprintf(p.gp, "%f %f\n", th, param.getVal("swft_r", "max"));
    }
    fprintf(p.gp, "%f %f\n", th, param.getVal("swft_r", "min"));
    fprintf(p.gp, "e\n");
    // icp
    fprintf(p.gp, "%f %f\n", state.icp.th, state.icp.r);
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
      fprintf(p.gp, "%f %f\n", (foot_l[i] + state.swft).th,
              (foot_l[i] + state.swft).r);
    }
    fprintf(p.gp, "e\n");
    // capture region
    std::vector<CaptureSet> region = capturability.getCaptureRegion(state, 0);
    if (!region.empty()) {
      for (size_t i = 0; i < region.size(); i++) {
        fprintf(p.gp, "%f %f\n", region[i].swft.th, region[i].swft.r);
      }
    }
    fprintf(p.gp, "e\n");

    // flush
    fflush(p.gp);

    state.icp.th += param.getVal("icp_th", "step");
  }
}

void CRPlot::plotCaptureIcp() {
  Grid grid(param);

  State state;
  state.icp.setPolar(0, 0);
  state.swft.setPolar(0.1, 120 * 3.14159 / 180);

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
  std::vector<Vector2> foot_r = model.getVec("foot", "foot_r_convex");
  for (size_t i = 0; i < foot_r.size(); i++) {
    fprintf(p.gp, "%f %f\n", foot_r[i].th, foot_r[i].r);
  }
  fprintf(p.gp, "e\n");
  // swing foot
  std::vector<Vector2> foot_l = model.getVec("foot", "foot_l_convex");
  for (size_t i = 0; i < foot_l.size(); i++) {
    fprintf(p.gp, "%f %f\n", (foot_l[i] + state.swft).th,
            (foot_l[i] + state.swft).r);
  }
  fprintf(p.gp, "e\n");
  // capturable ICP
  float icp_r = param.getVal("icp_r", "min");
  float icp_th = param.getVal("icp_th", "min");
  while (icp_r < param.getVal("icp_r", "max")) {
    icp_th = param.getVal("icp_th", "min");
    while (icp_th < param.getVal("icp_th", "max")) {
      state.icp.setPolar(icp_r, icp_th);
      std::vector<CaptureSet> region = capturability.getCaptureRegion(state, 0);
      if (!region.empty()) {
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
}