#include "capturability.h"
#include "cr_plot.h"
#include "gnuplot.h"
#include "grid.h"
#include "model.h"
#include "param.h"
#include "state.h"
#include <stdlib.h>
#include <vector>

using namespace std;
using namespace Capt;

class AnimCop : public Gnuplot {
public:
  AnimCop(Model model, Param param) {
    this->model = &model;
    this->param = &param;

    p("set title \"Capture Region (polar coordinate)\"");
    p("set encoding utf8");
    p("set size square");
    // p("unset key");

    p("set xtics 0.05");
    p("set ytics 0.05");

    std::string step_r = std::to_string(param.getVal("swft_r", "step"));
    std::string step_th = std::to_string(param.getVal("swft_th", "step"));

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

    p("set terminal gif animate optimize delay 5 size 600,900");
    p("set output 'plot.gif'");

    foot_r = model.getVec("foot", "foot_r_convex");

    FILE *fp;
    double buf[2];
    if ((fp = fopen("cop_list.csv", "r")) == NULL) {
      printf("Error: Couldn't find the file cop_list.csv\n");
      exit(EXIT_FAILURE);
    } else {
      int i = 0;
      while (fscanf(fp, "%lf,%lf", &buf[0], &buf[1]) != EOF) {
        cop[i].setCartesian(buf[0], buf[1]);
        i++;
      }
    }
    fclose(fp);
  }

  ~AnimCop(){};

  void plot(State state, int state_id) {
    // setting
    fprintf(p.gp, "plot ");
    // support foot
    fprintf(p.gp, "'-' t 'Current Support Foot' with lines linewidth 2 "
                  "lc \"black\",");
    // icp
    fprintf(p.gp, "'-' t 'Instantaneous Capture Point' with points pointsize 1 "
                  "pointtype 7 lc \"blue\",");
    // cop
    fprintf(p.gp, "'-' t 'Center of Pressure' with points pointsize 1 "
                  "pointtype 7 lc \"red\"\n");

    // plot
    // support foot
    for (size_t i = 0; i < foot_r.size(); i++) {
      fprintf(p.gp, "%f %f\n", foot_r[i].th, foot_r[i].r);
    }
    fprintf(p.gp, "e\n");
    // icp
    fprintf(p.gp, "%f %f\n", state.icp.th, state.icp.r);
    fprintf(p.gp, "e\n");
    // cop
    fprintf(p.gp, "%f %f\n", cop[state_id].th, cop[state_id].r);
    fprintf(p.gp, "e\n");

    // flush
    fflush(p.gp);
  }

private:
  Model *model;
  Param *param;
  Gnuplot p;

  std::vector<Vector2> foot_r;
  Vector2 cop[27195];
};

int main() {
  Param param("analysis.xml");
  Model model("nao.xml");
  Grid grid(param);
  AnimCop anim(model, param);
  State state;

  for (int state_id = 0; state_id < grid.getNumState(); state_id++) {
    state = grid.getState(state_id);
    anim.plot(state, state_id);
  }

  return 0;
}