#include "plot_trajectory.h"

using namespace std;

namespace CA {

PlotTrajectory::PlotTrajectory(Model model, Param param)
    : model(model), param(param), planning(model, param) {
  string output = "file";

  p("set title \"Reference Trajectory\"");
  p("set encoding utf8");
  p("set size square");
  // p("unset key");

  p("set xtics 0.1");
  p("set mxtics 10");
  p("set ytics 0.1");
  p("set mytics 10");
  p("set xrange [-0.1:0.3]");
  p("set yrange [0:0.4]");
  p("set grid mxtics mytics");

  if (output == "file") {
    p("set terminal gif animate optimize delay 10 size 600,900");
    p("set output 'plot.gif'");
  }
}

PlotTrajectory::~PlotTrajectory() {}

void PlotTrajectory::setCom(vec3_t com) { planning.setCom(com); }

void PlotTrajectory::setComVel(vec3_t com_vel) { planning.setComVel(com_vel); }

void PlotTrajectory::setRLeg(vec3_t rleg) { planning.setRLeg(rleg); }

void PlotTrajectory::setLLeg(vec3_t lleg) { planning.setLLeg(lleg); }

void PlotTrajectory::calcRef() { planning.calcRef(); }

void PlotTrajectory::plot(float dt) {
  // setting
  fprintf(p.gp, "plot ");
  // ICP
  fprintf(p.gp, "'-' t 'ICP' with points pointsize 1.5 "
                "pointtype 7 lc \"red\",");
  // Cop
  fprintf(p.gp, "'-' t 'Cop' with points pointsize 1.5 "
                "pointtype 7 lc \"orange\",");
  // Lod
  fprintf(p.gp, "'-' with lines linewidth 1 "
                "lc \"black\",");
  // Com
  fprintf(p.gp, "'-' t 'Com' with points pointsize 1.5 "
                "pointtype 7 lc \"blue\",");
  // ComVel
  fprintf(p.gp, "'-' t 'Com Velocity' with lines linewidth 3 "
                "lc \"blue\",");
  // Torso
  fprintf(p.gp, "'-' t 'Torso' with lines linewidth 1 "
                "lc \"black\",");
  // Right Leg
  fprintf(p.gp, "'-' t 'Right Leg' with lines linewidth 1 "
                "lc \"black\",");
  // Left Leg
  fprintf(p.gp, "'-' t 'Left Leg' with lines linewidth 1 "
                "lc \"black\"\n");

  world_p_icp = planning.getIcp(dt);
  world_p_cop = planning.getCop(dt);
  world_p_com = planning.getCom(dt);
  world_p_com_vel = planning.getComVel(dt);
  world_p_rleg = planning.getRLeg(dt);
  world_p_lleg = planning.getLLeg(dt);

  // plot (y, z)
  // ICP
  fprintf(p.gp, "%f %f\n", world_p_icp.y, 0.0);
  fprintf(p.gp, "e\n");
  // Cop
  fprintf(p.gp, "%f %f\n", world_p_cop.y, 0.0);
  fprintf(p.gp, "e\n");
  // Lod
  fprintf(p.gp, "%f %f\n", world_p_cop.y, 0.0);
  fprintf(p.gp, "%f %f\n", world_p_com.y(), world_p_com.z());
  fprintf(p.gp, "e\n");
  // Com
  fprintf(p.gp, "%f %f\n", world_p_com.y(), world_p_com.z());
  fprintf(p.gp, "e\n");
  // ComVel
  fprintf(p.gp, "%f %f\n", world_p_com.y(), world_p_com.z());
  fprintf(p.gp, "%f %f\n", world_p_com.y() + world_p_com_vel.y() * 0.05,
          world_p_com.z());
  fprintf(p.gp, "e\n");
  // Torso
  fprintf(p.gp, "e\n");
  // RLeg
  fprintf(p.gp, "%f %f\n", world_p_rleg.y() + 0.035, world_p_rleg.z());
  fprintf(p.gp, "%f %f\n", world_p_rleg.y() + 0.035, world_p_rleg.z() + 0.045);
  fprintf(p.gp, "%f %f\n", world_p_rleg.y() - 0.035, world_p_rleg.z() + 0.045);
  fprintf(p.gp, "%f %f\n", world_p_rleg.y() - 0.035, world_p_rleg.z());
  fprintf(p.gp, "%f %f\n", world_p_rleg.y() + 0.035, world_p_rleg.z());
  fprintf(p.gp, "e\n");
  // LLeg
  fprintf(p.gp, "%f %f\n", world_p_lleg.y() + 0.035, world_p_lleg.z());
  fprintf(p.gp, "%f %f\n", world_p_lleg.y() + 0.035, world_p_lleg.z() + 0.045);
  fprintf(p.gp, "%f %f\n", world_p_lleg.y() - 0.035, world_p_lleg.z() + 0.045);
  fprintf(p.gp, "%f %f\n", world_p_lleg.y() - 0.035, world_p_lleg.z());
  fprintf(p.gp, "%f %f\n", world_p_lleg.y() + 0.035, world_p_lleg.z());
  fprintf(p.gp, "e\n");

  // flush
  fflush(p.gp);
}
}