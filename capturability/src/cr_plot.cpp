#include "cr_plot.h"

using namespace std;

namespace CA {

CRPlot::CRPlot(Model model, Param param, std::string output)
    : model(model), param(param) {
  this->output = output;

  p("set title \"Capture Region (polar coordinate)\"");
  p("set encoding utf8");
  p("set size square");
  // p("unset key");

  string limit = "[";
  limit += std::to_string(0);
  limit += ":";
  limit += std::to_string(param.getVal("swft_r", "max") + 0.05);
  limit += "]";
  p("set xtics 0.05");
  p("set ytics 0.05");
  // p("set yrange " + limit);

  p("set polar");
  p("set theta top");
  p("set theta counterclockwise");
  p("set grid polar " + std::to_string(param.getVal("swft_th", "step")));
  p("set rrange " + limit);
  p("set trange [0:6.28]");
  p("set rtics scale 0");
  p("set rtics " + std::to_string(param.getVal("swft_r", "step")));
  p("set rtics format \"\"");
  p("unset raxis");

  if (output == "gif") {
    p("set terminal gif animate optimize delay 50 size 600,900");
    p("set output 'plot.gif'");
  }
  if (output == "svg") {
    p("set terminal svg");
    p("set output 'plot.svg'");
  }
}

CRPlot::~CRPlot() {}

void CRPlot::plot() {
  p("plot '-'\n");
  p("0.1 0.1\n");
  p("e\n");
};

void CRPlot::plot(State state, std::vector<CaptureSet> region) {
  // setting
  fprintf(p.gp, "plot ");
  // steppable region
  fprintf(p.gp, "'-' t 'Valid Stepping Region' with lines linewidth 1 "
                "lc \"black\",");
  // icp
  fprintf(p.gp, "'-' t 'Instantaneous Capture Point' with points pointsize 1 "
                "pointtype 7 lc \"blue\",");
  // swing foot
  fprintf(p.gp, "'-' t 'Current Swing Foot' with lines linewidth 2 "
                "lc \"black\",");
  // support foot
  fprintf(p.gp, "'-' t 'Current Support Foot' with lines linewidth 2 "
                "lc \"black\",");
  // capture region
  int step_num = 1;
  fprintf(p.gp,
          "'-' t '%d-step' with points pointsize 1 pointtype 7 lc \"%s\"\n",
          step_num, "gray");
  // step_num, p.int_color[step_num].c_str());

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
  // swing foot
  std::vector<Vector2> polygon = model.getVec("foot", "foot_l");
  for (size_t i = 0; i < polygon.size(); i++) {
    fprintf(p.gp, "%f %f\n", (polygon[i] + state.swft).th,
            (polygon[i] + state.swft).r);
  }
  fprintf(p.gp, "e\n");
  // support foot
  polygon = model.getVec("foot", "foot_r");
  for (size_t i = 0; i < polygon.size(); i++) {
    fprintf(p.gp, "%f %f\n", polygon[i].th, polygon[i].r);
  }
  fprintf(p.gp, "e\n");
  // capture region
  if (!region.empty()) {
    for (size_t i = 0; i < region.size(); i++) {
      fprintf(p.gp, "%f %f\n", region[i].swft.th, region[i].swft.r);
    }
  }
  fprintf(p.gp, "e\n");

  // flush
  fflush(p.gp);
}
}