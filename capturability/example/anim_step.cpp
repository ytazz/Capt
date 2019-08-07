#include "Capt.h"
#include "gnuplot.h"
#include <stdlib.h>
#include <vector>

using namespace std;
using namespace Capt;

class AnimStep : public Gnuplot {
public:
  AnimStep(Model model, Param param) : model(model), param(param), swing_foot(model), pendulum(model) {
    p("set title \"Dynamics (polar coordinate)\"");
    p("set encoding utf8");
    p("set size square");
    // p("unset key");

    p("set xtics 0.05");
    p("set ytics 0.05");

    std::string step_r  = std::to_string(param.getVal("swft_r", "step"));
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

    p("set terminal gif animate optimize delay 10 size 600,900");
    p("set output 'plot.gif'");
  }

  ~AnimStep(){
  };

  void plot(State state, Input input) {
    swing_foot.set(state.swft, input.swft);
    double  step_time=swing_foot.getTime();
    Polygon polygon;
    Vector2 cop = polygon.getClosestPoint(state.icp, model.getVec("foot", "foot_r_convex"));
    pendulum.setCop(cop);
    pendulum.setIcp(state.icp);

    std::vector<Vector2> foot_r;
    std::vector<Vector2> foot_l;

    double t=0.0;
    while(t<=step_time) {
      Vector2 swft, icp;
      swft.setCartesian(swing_foot.getTraj(t).x(), swing_foot.getTraj(t).y());
      icp = pendulum.getIcp(t);

      // setting
      fprintf(p.gp, "plot ");
      // support foot
      fprintf(p.gp, "'-' t 'Current Support Foot' with lines linewidth 2 "
              "lc \"black\",");
      // swing foot
      fprintf(p.gp, "'-' t 'Current Swing Foot' with lines linewidth 2 "
              "lc \"black\",");
      // icp
      fprintf(p.gp, "'-' t 'Instantaneous Capture Point' with points pointsize 1 "
              "pointtype 7 lc \"blue\",");
      // cop
      fprintf(p.gp, "'-' t 'Center of Pressure' with points pointsize 1 "
              "pointtype 7 lc \"red\"\n");

      // plot
      // support foot
      foot_r = model.getVec("foot", "foot_r_convex");
      for (size_t i = 0; i < foot_r.size(); i++) {
        fprintf(p.gp, "%f %f\n", foot_r[i].th, foot_r[i].r);
      }
      fprintf(p.gp, "e\n");
      // swing foot
      foot_l = model.getVec("foot", "foot_l_convex", swft);
      for (size_t i = 0; i < foot_l.size(); i++) {
        fprintf(p.gp, "%f %f\n", foot_l[i].th, foot_l[i].r);
      }
      fprintf(p.gp, "e\n");
      // icp
      fprintf(p.gp, "%f %f\n", icp.th, icp.r);
      fprintf(p.gp, "e\n");
      // cop
      fprintf(p.gp, "%f %f\n", cop.th, cop.r);
      fprintf(p.gp, "e\n");

      // flush
      fflush(p.gp);

      t+=0.01;
    }

    p("set terminal svg");
    p("set output 'plot.svg'");

    std::vector<Vector2> foot_convex;

    Vector2 swft, icp;
    swft.setCartesian(swing_foot.getTraj(step_time).x(), swing_foot.getTraj(step_time).y());
    icp = pendulum.getIcp(step_time);

    // setting
    fprintf(p.gp, "plot ");
    // support foot
    fprintf(p.gp, "'-' t 'Current Support Foot' with lines linewidth 2 "
            "lc \"gray\",");
    // swing foot
    fprintf(p.gp, "'-' t 'Current Swing Foot' with lines linewidth 2 "
            "lc \"gray\",");
    // swing foot
    fprintf(p.gp, "'-' t 'Foot Convex' with lines linewidth 2 "
            "lc \"black\",");
    // icp
    fprintf(p.gp, "'-' t 'Instantaneous Capture Point' with points pointsize 0.5 "
            "pointtype 7 lc \"blue\",");
    // cop
    fprintf(p.gp, "'-' t 'Center of Pressure' with points pointsize 0.5 "
            "pointtype 7 lc \"red\"\n");

    // plot
    // support foot
    foot_r = model.getVec("foot", "foot_r_convex");
    for (size_t i = 0; i < foot_r.size(); i++) {
      fprintf(p.gp, "%f %f\n", foot_r[i].th, foot_r[i].r);
    }
    fprintf(p.gp, "e\n");
    // swing foot
    foot_l = model.getVec("foot", "foot_l_convex", swft);
    for (size_t i = 0; i < foot_l.size(); i++) {
      fprintf(p.gp, "%f %f\n", foot_l[i].th, foot_l[i].r);
    }
    fprintf(p.gp, "e\n");
    // foot convex
    polygon.setVertex(foot_r);
    polygon.setVertex(foot_l);
    foot_convex=polygon.getConvexHull();
    for (size_t i = 0; i < foot_convex.size(); i++) {
      fprintf(p.gp, "%f %f\n", foot_convex[i].th, foot_convex[i].r);
    }
    fprintf(p.gp, "e\n");
    // icp
    fprintf(p.gp, "%f %f\n", icp.th, icp.r);
    fprintf(p.gp, "e\n");
    // cop
    fprintf(p.gp, "%f %f\n", cop.th, cop.r);
    fprintf(p.gp, "e\n");

    // flush
    fflush(p.gp);

  }

private:
  Model   model;
  Param   param;
  Gnuplot p;

  SwingFoot swing_foot;
  Pendulum  pendulum;
};

int main() {
  Param    param("analysis.xml");
  Model    model("nao.xml");
  Grid     grid(param);
  AnimStep anim(model, param);

  int state_id=0;
  int input_id=0;

  std::cout << "state_id: ";
  std::cin >> state_id;
  std::cout << "input_id: ";
  std::cin >> input_id;

  State state = grid.getState(state_id);
  Input input = grid.getInput(input_id);

  anim.plot(state, input);

  return 0;
}