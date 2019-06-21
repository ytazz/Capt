#include "planning.h"

namespace CA {

Planning::Planning(Model model, Param param)
    : polygon(), pendulum(model), capturability(model, param), grid(param),
      swing_foot(model) {
  this->com = vec3_t::Zero();
  this->com_vel = vec3_t::Zero();
  this->rleg = vec3_t::Zero();
  this->lleg = vec3_t::Zero();
  this->com_ref = vec3_t::Zero();
  this->com_vel_ref = vec3_t::Zero();
  this->rleg_ref = vec3_t::Zero();
  this->lleg_ref = vec3_t::Zero();

  g = model.getVal("environment", "gravity");
  h = model.getVal("physics", "com_height");
  omega = sqrt(g / h);

  capturability.load("1step.csv");
}

Planning::~Planning() {}

void Planning::setCom(vec3_t com) { this->com = com; }

void Planning::setComVel(vec3_t com_vel) { this->com_vel = com_vel; }

void Planning::setRLeg(vec3_t rleg) { this->rleg = rleg; }

void Planning::setLLeg(vec3_t lleg) { this->lleg = lleg; }

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

void Planning::calcRef() {
  // get capture region
  vec3_t icp3 = com + com_vel / omega;

  State state;
  state.icp.setCartesian(icp3.x(), icp3.y());
  state.swft.setCartesian((lleg - rleg).x(), (lleg - rleg).y());
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
  std::cout << "landing" << '\n';
  std::cout << landing << '\n';

  // feet trajectory planning
  swing_foot.set(lleg, landing);
  step_time = swing_foot.getTime();

  // com trajectory planning
  pendulum.setCom(com);
  pendulum.setComVel(com_vel);
  pendulum.setCop(capture_set[dist_min_id].cop);
}

vec3_t Planning::getCom(float time) {
  vec2_t com2_ref;
  if (time <= step_time)
    com2_ref = pendulum.getCom(time);
  else
    com2_ref = pendulum.getCom(step_time);
  com_ref = vec2tovec3(com2_ref);
  com_ref.z() = h;
  return com_ref;
}

vec3_t Planning::getRLeg(float time) {
  rleg_ref = rleg;
  return rleg_ref;
}

vec3_t Planning::getLLeg(float time) {
  if (time <= step_time)
    lleg_ref = swing_foot.getTraj(time);
  return lleg_ref;
}

} // namespace CA