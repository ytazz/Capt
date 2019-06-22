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
  p("set yrange [-0.1:0.3]");
  p("set grid mxtics mytics");

  if (output == "file") {
    p("set terminal gif animate optimize delay 10 size 600,900");
    p("set output 'plot.gif'");
  }

  fp = fopen("trajectory.csv", "w");
  fprintf(fp, "time,");
  fprintf(fp, "icp_x,icp_y,");
  fprintf(fp, "cop_x,cop_y,");
  fprintf(fp, "com_x,com_y,com_z,");
  fprintf(fp, "com_vel_x,com_vel_y,com_vel_z,");
  fprintf(fp, "rleg_x,rleg_y,rleg_z,");
  fprintf(fp, "lleg_x,lleg_y,lleg_z\n");
}

PlotTrajectory::~PlotTrajectory() {}

void PlotTrajectory::setCom(vec3_t com) { planning.setCom(com); }

void PlotTrajectory::setComVel(vec3_t com_vel) { planning.setComVel(com_vel); }

void PlotTrajectory::setRLeg(vec3_t rleg) { planning.setRLeg(rleg); }

void PlotTrajectory::setLLeg(vec3_t lleg) { planning.setLLeg(lleg); }

void PlotTrajectory::calcRef() { planning.calcRef(); }

void PlotTrajectory::fileOutput(vec2_t vec) {
  fprintf(fp, "%f,%f,", vec.x, vec.y);
}

void PlotTrajectory::fileOutput(vec3_t vec) {
  fprintf(fp, "%f,%f,%f,", vec.x(), vec.y(), vec.z());
}

void PlotTrajectory::plotYZ(float dt) {
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

  fprintf(fp, "%f,", dt);
  fileOutput(world_p_icp);
  fileOutput(world_p_cop);
  fileOutput(world_p_com);
  fileOutput(world_p_com_vel);
  fileOutput(world_p_rleg);
  fileOutput(world_p_lleg);
  fprintf(fp, "\n");

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

void PlotTrajectory::plotXY(float dt) {
  p("set view 60, 30, 1, 1");
  // setting
  fprintf(p.gp, "plot ");
  // ICP
  fprintf(p.gp, "'-' t 'ICP' with points pointsize 1.5 "
                "pointtype 7 lc \"red\",");
  // Cop
  fprintf(p.gp, "'-' t 'Cop' with points pointsize 1.5 "
                "pointtype 7 lc \"orange\",");
  // Com
  fprintf(p.gp, "'-' t 'Com' with points pointsize 1.5 "
                "pointtype 7 lc \"blue\",");
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

  fprintf(fp, "%f,", dt);
  fileOutput(world_p_icp);
  fileOutput(world_p_cop);
  fileOutput(world_p_com);
  fileOutput(world_p_com_vel);
  fileOutput(world_p_rleg);
  fileOutput(world_p_lleg);
  fprintf(fp, "\n");

  // plot (y, z)
  // ICP
  fprintf(p.gp, "%f %f\n", world_p_icp.x, world_p_icp.y);
  fprintf(p.gp, "e\n");
  // Cop
  fprintf(p.gp, "%f %f\n", world_p_cop.x, world_p_cop.y);
  fprintf(p.gp, "e\n");
  // Com
  fprintf(p.gp, "%f %f\n", world_p_com.x(), world_p_com.y());
  fprintf(p.gp, "e\n");
  // RLeg
  foot = model.getVec("foot", "foot_r");
  for (size_t i = 0; i < foot.size(); i++) {
    fprintf(p.gp, "%f %f\n", world_p_rleg.x() + foot[i].x,
            world_p_rleg.y() + foot[i].y);
  }
  fprintf(p.gp, "e\n");
  // LLeg
  foot = model.getVec("foot", "foot_l");
  for (size_t i = 0; i < foot.size(); i++) {
    fprintf(p.gp, "%f %f\n", world_p_lleg.x() + foot[i].x,
            world_p_lleg.y() + foot[i].y);
  }
  fprintf(p.gp, "e\n");

  // flush
  fflush(p.gp);
}
}