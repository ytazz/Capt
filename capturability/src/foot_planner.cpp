#include "foot_planner.h"

namespace CA {

FootPlanner::FootPlanner(Model *model, Capturability *capturability, Grid *grid)
    : model(model), capturability(capturability), grid(grid) {
  num_step = 0;

  g = model->getVal("environment", "gravity");
  h = model->getVal("physics", "com_height");
  omega = sqrt(g / h);
}

FootPlanner::~FootPlanner() {}

void FootPlanner::setComState(vec3_t com, vec3_t com_vel) {
  this->com_cr = com;
  this->com_vel_cr = com_vel;

  this->icp_cr.setCartesian(com.x() + com_vel.x() / omega,
                            com.y() + com_vel.y() / omega);
}

void FootPlanner::setRLeg(vec3_t rleg) { this->foot_step.foot_r_ini = rleg; }

void FootPlanner::setLLeg(vec3_t lleg) { this->foot_step.foot_l_ini = lleg; }

GridState FootPlanner::calcCurrentState(SupportFoot suft) {
  State state;
  GridState gstate;

  vec2_t suft_p_swft; // support foot coord swing foot pos
  vec2_t suft_p_icp;  // support foot coord icp pos
  if (foot_step.suft[0] == SUPPORT_R) {
    suft_p_icp.setCartesian(icp_cr.x - foot_step.foot_r_ini.x(),
                            icp_cr.y - foot_step.foot_r_ini.y());
    suft_p_swft.setCartesian(
        foot_step.foot_l_ini.x() - foot_step.foot_r_ini.x(),
        foot_step.foot_l_ini.y() - foot_step.foot_r_ini.y());
  }
  if (foot_step.suft[0] == SUPPORT_L) {
    suft_p_icp.setCartesian(icp_cr.x - foot_step.foot_l_ini.x(),
                            -(icp_cr.y - foot_step.foot_l_ini.y()));
    suft_p_swft.setCartesian(
        foot_step.foot_r_ini.x() - foot_step.foot_l_ini.x(),
        -(foot_step.foot_r_ini.y() - foot_step.foot_l_ini.y()));
  }
  state.icp = suft_p_icp;
  state.swft = suft_p_swft;
  gstate = grid->roundState(state);

  return gstate;
}

void FootPlanner::getInitState() {
  int n_step = 1;
  while (true) {
    if (foot_step.suft.empty())
      foot_step.suft.push_back(SUPPORT_R);
    else if (foot_step.suft[0] == SUPPORT_R)
      foot_step.suft[0] = SUPPORT_L;
    else
      foot_step.suft[0] = SUPPORT_R;

    GridState gstate = calcCurrentState(foot_step.suft[0]);
    capture_set = capturability->getCaptureRegion(gstate.id, n_step);

    if (!capture_set.empty())
      break;

    if (foot_step.suft[0] == SUPPORT_L)
      n_step++;
    if (n_step >= 5) // 5 = max n-step capturable
      break;
  }
}

bool FootPlanner::plan() {
  bool flag = true;

  // calculate initial support foot and
  // capture region
  getInitState();

  // calculate desired landing position
  vec3_t landing;
  float dist = 0.0, dist_min = 0.0;
  int dist_min_id = 0;
  GridState gstate = calcCurrentState(foot_step.suft[0]);
  std::cout << "size = " << capture_set.size() << '\n';
  for (size_t i = 0; i < capture_set.size(); i++) {
    dist = (capture_set[i].swft - gstate.state.swft).norm();
    if (i == 0) {
      dist_min = dist;
      dist_min_id = i;
    } else if (dist < dist_min) {
      dist_min = dist;
      dist_min_id = i;
    }
  }
  landing.x() = capture_set[dist_min_id].swft.x;
  landing.y() = capture_set[dist_min_id].swft.y;
  landing.z() = 0.0;
  std::cout << "landing no." << dist_min_id << '\n';
  std::cout << landing << '\n';

  // set footstep
  vec3_t foot;
  vec3_t cop;
  if (foot_step.suft[0] == SUPPORT_R) {
    foot = foot_step.foot_r_ini + landing;
    cop.x() = foot_step.foot_r_ini.x() + capture_set[dist_min_id].cop.x;
    cop.y() = foot_step.foot_r_ini.y() + capture_set[dist_min_id].cop.y;
    cop.z() = 0.0;
  }
  if (foot_step.suft[0] == SUPPORT_L) {
    foot.x() = foot_step.foot_l_ini.x() + landing.x();
    foot.y() = foot_step.foot_l_ini.y() - landing.y();
    foot.z() = 0.0;
    cop.x() = foot_step.foot_l_ini.x() + capture_set[dist_min_id].cop.x;
    cop.y() = foot_step.foot_l_ini.y() - capture_set[dist_min_id].cop.y;
    cop.z() = 0.0;
  }
  foot_step.foot.push_back(foot);
  foot_step.cop.push_back(cop);
  foot_step.step_time.push_back(capture_set[dist_min_id].step_time);

  return flag;
}

FootStep FootPlanner::getFootStep() { return foot_step; }

void FootPlanner::show() {
  printf("|no.\t|leg\t|x\t|y\t|z\t|t\t|\n");
  for (size_t i = 0; i < foot_step.size(); i++) {
    printf("|%d\t|%d\t", i, foot_step.suft[i]);
    printf("|%1.3f\t|%1.3f\t|%1.3f\t", foot_step.foot[i].x(),
           foot_step.foot[i].y(), foot_step.foot[i].z());
    printf("|%1.3f\t|\n", foot_step.step_time[i]);
  }
}
} // namespace CA