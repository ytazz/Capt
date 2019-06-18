#include "monitor.h"

namespace CA {

Monitor::Monitor(Model model) : model(model), pendulum(model), polygon() {}

Monitor::~Monitor() {}

void Monitor::setSuft(vec2_t suft, EFoot which_suft) {
  this->suft = suft;
  this->e_suft = which_suft;
}

void Monitor::setIcp(vec2_t icp) {
  this->icp = icp;

  polygon.clear();
  switch (e_suft) {
  case R_FOOT:
    polygon.setVertex(model.getVec("foot", "foot_r"));
    break;
  case L_FOOT:
    polygon.setVertex(model.getVec("foot", "foot_l"));
    break;
  default:
    printf("Error: support foot is invalid\n");
    exit(EXIT_FAILURE);
    break;
  }
  std::vector<vec2_t> support_region = polygon.getConvexHull();

  this->cop = polygon.getClosestPoint(icp, support_region);

  pendulum.setIcp(this->icp);
  pendulum.setCop(this->cop);
}

void Monitor::setLandingPos(vec2_t landing_pos) {
  this->landing_pos = landing_pos;
}

void Monitor::setStepTime(float step_time) { this->step_time = step_time; }

bool Monitor::judge() {
  vec2_t icp_ = pendulum.getIcp(step_time);

  polygon.clear();
  switch (e_suft) {
  case R_FOOT:
    polygon.setVertex(model.getVec("foot", "foot_l"));
    break;
  case L_FOOT:
    polygon.setVertex(model.getVec("foot", "foot_r"));
    break;
  default:
    printf("Error: support foot is invalid\n");
    exit(EXIT_FAILURE);
    break;
  }
  std::vector<vec2_t> landing_region = polygon.getConvexHull();

  bool flag = polygon.inPolygon(icp_, landing_region);
  return flag;
}

} // namespace CA