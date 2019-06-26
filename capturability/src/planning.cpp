#include "planning.h"

namespace CA {

Planning::Planning(Model model, Param param, float timestep)
    : model(model), polygon(), pendulum(model), capturability(model, param),
      grid(param), swing_foot(model), k(0.5), dt(timestep) {
  this->step_time = 0.0;

  this->cop_cmd.clear();
  this->com_cmd = vec3_t::Zero();
  this->rleg_cmd = vec3_t::Zero();
  this->lleg_cmd = vec3_t::Zero();

  this->cop_0_des.clear();
  this->icp_0_des.clear();
  this->com_0_des.clear();
  this->com_vel_0_des.clear();
  this->rleg_des.clear();
  this->lleg_des.clear();

  g = model.getVal("environment", "gravity");
  h = model.getVal("physics", "com_height");
  omega = sqrt(g / h);
  std::cout << "omega" << omega << '\n';

  capturability.load("1step.csv");
}

Planning::~Planning() {}

void Planning::setCom(vec3_t com) { this->com_cr = com; }

void Planning::setComVel(vec3_t com_vel) { this->com_vel_cr = com_vel; }

void Planning::setIcp(vec2_t icp) { this->icp_cr = icp; }

void Planning::setRLeg(vec3_t rleg) { this->rleg_des.push_back(rleg); }

void Planning::setLLeg(vec3_t lleg) { this->lleg_des.push_back(lleg); }

vec3_t Planning::vec2tovec3(vec2_t vec2) {
  vec3_t vec3;
  vec3.x() = vec2.x;
  vec3.y() = vec2.y;
  vec3.z() = 0.0;
  return vec3;
}

vec2_t Planning::vec3tovec2(vec3_t vec3) {
  vec2_t vec2;
  vec2.setCartesian(vec3.x(), vec3.y());
  return vec2;
}

void Planning::calcDes() {
  // get capture region
  vec3_t icp3 = com_cr + com_vel_cr / omega;
  std::cout << "com_cr" << '\n';
  std::cout << com_cr << '\n';
  std::cout << "com_vel_cr" << '\n';
  std::cout << com_vel_cr << '\n';

  State state;
  state.icp.setCartesian((icp3 - rleg_des[0]).x(), (icp3 - rleg_des[0]).y());
  state.swft.setCartesian((lleg_des[0] - rleg_des[0]).x(),
                          (lleg_des[0] - rleg_des[0]).y());
  state.printCartesian();

  GridState gstate;
  gstate = grid.roundState(state);

  std::vector<CA::CaptureSet> capture_set;
  capture_set = capturability.getCaptureRegion(gstate.id, 1);
  // printf("capture region\n");
  // for (size_t i = 0; i < capture_set.size(); i++) {
  //   printf("%f,%f\n", capture_set[i].swft.x, capture_set[i].swft.y);
  // }

  // calculate desired landing position
  vec3_t landing;
  float dist = 0.0, dist_min = 0.0;
  int dist_min_id = 0;
  std::cout << "size = " << capture_set.size() << '\n';
  for (size_t i = 0; i < capture_set.size(); i++) {
    dist = (capture_set[i].swft - state.swft).norm();
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
  lleg_des.push_back(rleg_des[0] + landing);

  // foot trajectory planning
  swing_foot.set(lleg_des[0], lleg_des[1]);
  step_time = swing_foot.getTime();

  // trajectory planning
  cop_0_des.resize(2);
  icp_0_des.resize(2);
  com_0_des.resize(2);
  com_vel_0_des.resize(2);
  // 0th step
  cop_0_des[0] = capture_set[dist_min_id].cop + vec3tovec2(rleg_des[0]);
  icp_0_des[0] = icp_cr;
  com_0_des[0] = com_cr;
  com_vel_0_des[0] = com_vel_cr;
  pendulum.setCop(cop_0_des[0]);
  pendulum.setIcp(icp_0_des[0]);
  pendulum.setCom(com_0_des[0]);
  pendulum.setComVel(com_vel_0_des[0]);
  cop_0_des[1] = pendulum.getIcp(step_time);
  icp_0_des[1] = pendulum.getIcp(step_time);
  com_0_des[1] = vec2tovec3(pendulum.getCom(step_time));
  com_vel_0_des[1] = vec2tovec3(pendulum.getComVel(step_time));
}

vec2_t Planning::getIcpDes(float time) {
  if (time < step_time) {
    pendulum.setCop(cop_0_des[0]);
    pendulum.setIcp(icp_0_des[0]);
    pendulum.setCom(com_0_des[0]);
    pendulum.setComVel(com_vel_0_des[0]);
    icp_des = pendulum.getIcp(time);
  } else {
    // icp_des = icp_0_des[1];
    icp_des.x = lleg_des[1].x();
    icp_des.y = lleg_des[1].y();
  }
  return icp_des;
}

vec2_t Planning::getIcpVelDes(float time) {
  if (time < step_time) {
    pendulum.setCop(cop_0_des[0]);
    pendulum.setIcp(icp_0_des[0]);
    pendulum.setCom(com_0_des[0]);
    pendulum.setComVel(com_vel_0_des[0]);
    icp_vel_des = pendulum.getIcpVel(time);
  } else {
    icp_vel_des.clear();
  }
  return icp_vel_des;
}

vec2_t Planning::getCop(float time) {
  vec2_t cop_cmd_;
  cop_cmd_ = icp_cr + k * (icp_cr - icp_des) / omega - icp_vel_des / omega;
  std::vector<vec2_t> region;
  if (time < step_time) {
    region = model.getVec("foot", "foot_r_convex", rleg_des[0]);
  } else {
    region = model.getVec("foot", "foot_l_convex", lleg_des[1]);
  }
  // polygon.setVertex(region);
  // region = polygon.getConvexHull();
  cop_cmd = polygon.getClosestPoint(cop_cmd_, region);
  cop_cmd = cop_cmd_;
  return cop_cmd;
}

vec3_t Planning::getCom(float time) {
  pendulum.setCom(com_cr);
  pendulum.setComVel(com_vel_cr);
  pendulum.setCop(cop_cmd);
  com_cmd.x() = pendulum.getCom(dt).x;
  com_cmd.y() = pendulum.getCom(dt).y;
  com_cmd.z() = h;
  return com_cmd;
}

vec3_t Planning::getComVel(float time) {
  pendulum.setCom(com_cr);
  pendulum.setComVel(com_vel_cr);
  pendulum.setCop(cop_cmd);
  com_vel_cmd.x() = pendulum.getComVel(dt).x;
  com_vel_cmd.y() = pendulum.getComVel(dt).y;
  com_vel_cmd.z() = 0.0;
  return com_vel_cmd;
}

vec2_t Planning::getIcp(float time) {
  pendulum.setIcp(icp_cr);
  pendulum.setCop(cop_cmd);
  icp_cmd = pendulum.getIcp(dt);
  return icp_cmd;
}

vec3_t Planning::getRLeg(float time) {
  rleg_cmd = rleg_des[0];
  if (time > step_time) {
    if (time <= 0.2) {
      rleg_cmd.y() = -0.05 + 0.1 * (time - step_time) / 0.5;
      rleg_cmd.z() = 0 + 0.2 * (time - step_time) / 0.5;
    } else {
      rleg_cmd.y() = -0.05 + 0.1 * (0.2 - step_time) / 0.5;
      rleg_cmd.z() = 0 + 0.2 * (0.2 - step_time) / 0.5;
    }
  }
  return rleg_cmd;
}

vec3_t Planning::getLLeg(float time) {
  if (time < step_time)
    lleg_cmd = swing_foot.getTraj(time);
  else
    lleg_cmd = lleg_des[1];
  return lleg_cmd;
}

} // namespace CA