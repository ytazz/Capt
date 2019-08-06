#include "friction_filter.h"

namespace Capt {

FrictionFilter::FrictionFilter(Capturability capturability, Pendulum pendulum)
    : capturability(capturability), pendulum(pendulum){};

FrictionFilter::~FrictionFilter(){};

void FrictionFilter::setCaptureRegion(std::vector<CaptureSet> capture_set) {
  for (size_t i = 0; i < capture_set.size(); i++) {
    this->capture_set.push_back(capture_set[i]);
  }
}

std::vector<CaptureSet>
FrictionFilter::getCaptureRegion(vec2_t com, vec2_t com_vel, float mu) {
  std::vector<CaptureSet> modified_cr; // cr = capture region

  std::vector<bool> flag;
  float d_start = 0, d_end = 0;
  vec2_t com_, com_vel_;
  for (size_t i = 0; i < capture_set.size(); i++) {
    // flag
    flag.clear();
    for (int i = 0; i <= capture_set[i].n_step; i++) {
      flag.push_back(false);
    }
    com_ = com;
    com_vel_ = com_vel;

    // 軌道初端での距離
    d_start = (com_ - capture_set[i].cop).norm();

    // 軌道終端での距離
    pendulum.setCop(capture_set[i].cop);
    pendulum.setCom(com_);
    pendulum.setComVel(com_vel_);
    com_ = pendulum.getCom(capture_set[i].step_time);
    com_vel_ = pendulum.getComVel(capture_set[i].step_time);
    d_end = (com_ - capture_set[i].cop).norm();

    if (std::max(d_start, d_end) <= mu * 0.25) {
      modified_cr.push_back(capture_set[i]);
    }
  }

  return modified_cr;
}

} // namespace Capt