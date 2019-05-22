#include "swing_foot.h"

namespace CA {

SwingFoot::SwingFoot(Model model) {
  foot.init();
  foot_des.init();
  foot_vel_x = 0;
  foot_vel_y = 0;

  foot_vel = model.getVal("physics", "foot_vel_max");
  step_time_min = model.getVal("physics", "step_time_min");
}

SwingFoot::~SwingFoot() {}

void SwingFoot::set(Vector2 foot, Vector2 foot_des) {
  this->foot = foot;
  this->foot_des = foot_des;

  float dx, dy;
  dx = foot_des.x - foot.x;
  dy = foot_des.y - foot.y;
  foot_vel_x = foot_vel * dx / sqrtf(dx * dx + dy * dy);
  foot_vel_y = foot_vel * dy / sqrtf(dx * dx + dy * dy);
}

float SwingFoot::getTime() {
  float t = (foot_des - foot).norm() / foot_vel + step_time_min;

  return t;
}

Vector2 SwingFoot::getTraj(float dt) {
  Vector2 swft;

  if (dt <= step_time_min / 2.0) {
    swft.x = foot.x;
    swft.y = foot.y;
  } else if (dt >= getTime() - step_time_min / 2.0) {
    swft.x = foot_des.x;
    swft.y = foot_des.y;
  } else {
    swft.x = foot.x + foot_vel_x * (dt - step_time_min / 2.0);
    swft.y = foot.y + foot_vel_y * (dt - step_time_min / 2.0);
  }

  return swft;
}

} // namespace CA
