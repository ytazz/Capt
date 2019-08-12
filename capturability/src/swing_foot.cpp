#include "swing_foot.h"

namespace Capt {

SwingFoot::SwingFoot(Model model) {
  foot       = Eigen::Vector3f::Zero();
  foot_des   = Eigen::Vector3f::Zero();
  foot_vel_x = 0.0;
  foot_vel_y = 0.0;

  foot_vel      = model.getVal("physics", "foot_vel_max");
  step_time_min = model.getVal("physics", "step_time_min");
  step_height   = model.getVal("physics", "step_height");
}

SwingFoot::~SwingFoot() {
}

void SwingFoot::set(vec2_t foot, vec2_t foot_des) {
  this->foot.x()     = foot.x;
  this->foot.y()     = foot.y;
  this->foot.z()     = 0.0;
  this->foot_des.x() = foot_des.x;
  this->foot_des.y() = foot_des.y;
  this->foot_des.z() = 0.0;

  set(this->foot, this->foot_des);
}

void SwingFoot::set(vec3_t foot, vec3_t foot_des) {
  this->foot.x()     = foot.x();
  this->foot.y()     = foot.y();
  this->foot.z()     = foot.z();
  this->foot_des.x() = foot_des.x();
  this->foot_des.y() = foot_des.y();
  this->foot_des.z() = foot_des.z();

  float dx, dy;
  dx         = foot_des.x() - foot.x();
  dy         = foot_des.y() - foot.y();
  foot_vel_x = foot_vel * dx / sqrtf(dx * dx + dy * dy);
  foot_vel_y = foot_vel * dy / sqrtf(dx * dx + dy * dy);

  interpolation[0].set(foot.x(), foot_des.x(), 0.0, 0.0, getTime() );
  interpolation[1].set(foot.y(), foot_des.y(), 0.0, 0.0, getTime() );
  interpolation[2].set(foot.z(), foot.z() + step_height, 0.0, 0.0,
                       getTime() / 2);
  interpolation[3].set(foot.z() + step_height, foot_des.z(), 0.0, 0.0,
                       getTime() / 2);
}

float SwingFoot::getTime() {
  float t = ( foot_des - foot ).norm() / foot_vel + step_time_min;

  return t;
}

vec3_t SwingFoot::getTraj(float dt) {
  vec3_t swft;

  // if (dt <= step_time_min / 2.0) {
  //   swft.x() = foot.x;
  //   swft.y() = foot.y;
  // } else if (dt >= getTime() - step_time_min / 2.0) {
  //   swft.x() = foot_des.x;
  //   swft.y() = foot_des.y;
  // } else {
  //   swft.x() = foot.x + foot_vel_x * (dt - step_time_min / 2.0);
  //   swft.y() = foot.y + foot_vel_y * (dt - step_time_min / 2.0);
  // }

  swft.x() = interpolation[0].get(dt);
  swft.y() = interpolation[1].get(dt);
  if (dt <= getTime() / 2.0) {
    swft.z() = interpolation[2].get(dt);
  } else {
    swft.z() = interpolation[3].get(dt - getTime() / 2.0);
  }

  return swft;
}

} // namespace Capt
