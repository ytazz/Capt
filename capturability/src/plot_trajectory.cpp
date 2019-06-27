#include "plot_trajectory.h"

using namespace std;

namespace CA {

PlotTrajectory::PlotTrajectory(Model model, Param param,
                               Capturability capturability, Grid grid,
                               float timestep)
    : model(model), param(param), planning(model, param, timestep),
      foot_planner(&model, capturability, &grid) {
  string output = "file";

  p("set title \"Reference Trajectory\"");
  p("set encoding utf8");
  p("set size square");
  // p("unset key");

  p("set xtics 0.1");
  p("set mxtics 10");
  p("set ytics 0.1");
  p("set mytics 10");
  p("set xrange [-0.2:0.2]");
  p("set yrange [-0.2:0.2]");
  p("set grid mxtics mytics");

  if (output == "file") {
    p("set terminal gif animate optimize delay 20 size 600,900");
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

void PlotTrajectory::plan(vec2_t icp, vec3_t com, vec3_t com_vel, vec3_t rleg,
                          vec3_t lleg) {
  foot_planner.setComState(com, com_vel);
  foot_planner.setRLeg(rleg);
  foot_planner.setLLeg(lleg);
  foot_planner.plan();
  foot_planner.show();

  planning.setIcp(icp);
  planning.setCom(com);
  planning.setComVel(com_vel);
  planning.setFootstep(foot_planner.getFootstep());
  planning.plan();
}

void PlotTrajectory::fileOutput(vec2_t vec) {
  fprintf(fp, "%f,%f,", vec.x, vec.y);
}

void PlotTrajectory::fileOutput(vec3_t vec) {
  fprintf(fp, "%f,%f,%f,", vec.x(), vec.y(), vec.z());
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

  world_p_cop = planning.getCop(dt);
  world_p_icp = planning.getIcp(dt);
  world_p_com = planning.getCom(dt);
  world_p_com_vel = planning.getComVel(dt);
  world_p_rleg = planning.getRLeg(dt);
  world_p_lleg = planning.getLLeg(dt);

  planning.setCom(world_p_com);
  // world_p_com_vel.y() += (5 - rand() % 11) * 0.005;
  // world_p_com_vel.x() += (5 - rand() % 11) * 0.005;
  planning.setComVel(world_p_com_vel);
  vec2_t icp;
  icp.setCartesian(world_p_com.x() + world_p_com_vel.x() / 6.26311,
                   world_p_com.y() + world_p_com_vel.y() / 6.26311);
  planning.setIcp(icp);

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
  foot = model.getVec("foot", "foot_r_convex");
  for (size_t i = 0; i < foot.size(); i++) {
    fprintf(p.gp, "%f %f\n", world_p_rleg.x() + foot[i].x,
            world_p_rleg.y() + foot[i].y);
  }
  fprintf(p.gp, "e\n");
  // LLeg
  foot = model.getVec("foot", "foot_l_convex");
  for (size_t i = 0; i < foot.size(); i++) {
    fprintf(p.gp, "%f %f\n", world_p_lleg.x() + foot[i].x,
            world_p_lleg.y() + foot[i].y);
  }
  fprintf(p.gp, "e\n");

  // flush
  fflush(p.gp);
}
} // namespace CA