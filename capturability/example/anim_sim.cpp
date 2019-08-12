#include "Capt.h"
#include "gnuplot.h"
#include <stdlib.h>
#include <vector>

using namespace std;
using namespace Capt;

class AnimSim : public Gnuplot {
public:
  AnimSim(Model model, Param param) : model(model), param(param) {
    p("set title \"Simulation Result\"");
    p("set encoding utf8");
    p("set size square");
    // p("unset key");

    p("set xtics 0.01");
    p("set ytics 0.01");

    p("set grid lt 1 lc \"gray\"");
    p("set xrange [-0.2:0.2]");
    p("set yrange [-0.2:0.2]");

    p("set terminal gif animate optimize delay 10 size 600,900");
    p("set output 'plot.gif'");
  }

  ~AnimSim(){
  };

  void plot() {
    std::vector<Vector2> foot_r;
    std::vector<Vector2> foot_l;

    double time;
    double cop_x, cop_y;
    double icp_x, icp_y;
    double com_ref_x, com_ref_y, com_ref_z;
    double com_x, com_y, com_z;
    double torso_ref_x, torso_ref_y, torso_ref_z;
    double torso_x, torso_y, torso_z;
    double rleg_ref_x, rleg_ref_y, rleg_ref_z;
    double rleg_x, rleg_y, rleg_z;
    double lleg_ref_x, lleg_ref_y, lleg_ref_z;
    double lleg_x, lleg_y, lleg_z;

    FILE *fp;
    if ( ( fp = fopen("/home/kuribayashi/choreonoid/build/bin/data.csv", "r") ) == NULL) {
      printf("Error: Couldn't find the file\n");
      exit(EXIT_FAILURE);
    } else {
      printf("Find data.\n");
      while (fscanf(fp,
                    "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,"
                    "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,"
                    "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                    &time,
                    &cop_x, &cop_y,
                    &icp_x, &icp_y,
                    &com_ref_x, &com_ref_y, &com_ref_z,
                    &com_x, &com_y, &com_z,
                    &torso_ref_x, &torso_ref_y, &torso_ref_z,
                    &torso_x, &torso_y, &torso_z,
                    &rleg_ref_x, &rleg_ref_y, &rleg_ref_z,
                    &rleg_x, &rleg_y, &rleg_z,
                    &lleg_ref_x, &lleg_ref_y, &lleg_ref_z,
                    &lleg_x, &lleg_y, &lleg_z
                    ) != EOF) {
        // plot
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
        vec2_t rleg_plot;
        rleg_plot.setCartesian(rleg_x, rleg_y);
        foot_r = model.getVec("foot", "foot_r_convex", rleg_plot);
        for (size_t i = 0; i < foot_r.size(); i++) {
          fprintf(p.gp, "%f %f\n", foot_r[i].x, foot_r[i].y);
        }
        fprintf(p.gp, "e\n");
        // swing foot
        vec2_t lleg_plot;
        lleg_plot.setCartesian(lleg_x, lleg_y);
        foot_l = model.getVec("foot", "foot_l_convex", lleg_plot);
        for (size_t i = 0; i < foot_l.size(); i++) {
          fprintf(p.gp, "%f %f\n", foot_l[i].x, foot_l[i].y);
        }
        fprintf(p.gp, "e\n");
        // icp
        fprintf(p.gp, "%f %f\n", icp_x, icp_y);
        fprintf(p.gp, "e\n");
        // cop
        fprintf(p.gp, "%f %f\n", cop_x, cop_y);
        fprintf(p.gp, "e\n");

        // flush
        fflush(p.gp);
      }
    }
  }

private:
  Model   model;
  Param   param;
  Gnuplot p;
};

int main() {
  Param   param("analysis.xml");
  Model   model("nao.xml");
  AnimSim anim(model, param);
  anim.plot();

  return 0;
}