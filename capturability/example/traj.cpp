#include "Capt.h"
#include "gnuplot.h"
#include <stdlib.h>
#include <vector>

using namespace std;
using namespace Capt;

class Traj : public Gnuplot {
public:
  Traj(){
    p("set title \"Simulation Result\"");
    p("set encoding utf8");
    p("set size square");
    // p("unset key");

    p("set xtics 0.05");
    p("set ytics 0.05");

    p("set grid lt 1 lc \"gray\"");
    p("set xrange [-0.1:0.1]");
    p("set yrange [0.0:0.2]");

    p("set terminal svg");
    p("set output 'plot.svg'");

    start.x() = 0.0;
    start.y() = 0.055;
    start.z() = 0.0;

    end.x() = 0.053;
    end.y() = 0.04;
    end.z() = 0.0;

    step_time = 0.1;

    cycloid.set(start, end, step_time);
  }

  ~Traj(){
  };

  void plot() {
    // plot
    fprintf(p.gp, "plot ");
    // support foot
    fprintf(p.gp, "'-' t 'x' with lines linewidth 2 lc \"red\",");
    fprintf(p.gp, "'-' t 'y' with lines linewidth 2 lc \"black\"\n");

    double t = 0.0;
    while(t < step_time) {
      vec3_t foot = cycloid.get(t);
      fprintf(p.gp, "%f %f\n", foot.x(), foot.z() );

      t+=0.001;
    }
    fprintf(p.gp, "e\n");

    t = 0.0;
    while(t < step_time) {
      vec3_t foot = cycloid.get(t);
      fprintf(p.gp, "%f %f\n", foot.y(), foot.z() );

      t+=0.001;
    }
    fprintf(p.gp, "e\n");

    fflush(p.gp);
  }

private:
  Gnuplot p;
  Cycloid cycloid;

  vec3_t start, end;
  double step_time;
};

int main() {
  Traj traj;
  traj.plot();

  return 0;
}