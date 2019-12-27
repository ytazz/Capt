#include "swing.h"

namespace Capt {

Swing::Swing(Model *model) {
  foot     = vec3_t::Zero();
  foot_des = vec3_t::Zero();

  step_time = 0.0;

  model->read(&foot_vel, "foot_vel_max");
  model->read(&step_time_min, "step_time_min");
  model->read(&step_height, "step_height");
}

Swing::~Swing() {
}

void Swing::set(vec2_t foot, vec2_t foot_des, double elapsed) {
  this->foot.x()     = foot.x();
  this->foot.y()     = foot.y();
  this->foot.z()     = 0.0;
  this->foot_des.x() = foot_des.x();
  this->foot_des.y() = foot_des.y();
  this->foot_des.z() = 0.0;

  set(this->foot, this->foot_des, elapsed);
}

void Swing::set(vec3_t foot, vec3_t foot_des, double elapsed) {
  this->foot.x()     = foot.x();
  this->foot.y()     = foot.y();
  this->foot.z()     = foot.z();
  this->foot_des.x() = foot_des.x();
  this->foot_des.y() = foot_des.y();
  this->foot_des.z() = foot_des.z();

  double dist = sqrt( ( foot_des.x() - foot.x() ) * ( foot_des.x() - foot.x() ) +
                      ( foot_des.y() - foot.y() ) * ( foot_des.y() - foot.y() ) );
  step_time = max(0, step_time_min / 2 - elapsed) + dist / foot_vel + step_time_min / 2;

  // cycloid.set(foot, foot_des, step_time);
  swingX.set(foot.x(), foot_des.x(), 0.0, step_time);
  swingY.set(foot.y(), foot_des.y(), 0.0, step_time);
  swingZ.set(foot.z(), foot_des.z(), 0.0, step_time);
}

double Swing::getTime() {
  return step_time;
}

vec3_t Swing::getTraj(double dt) {
  // return cycloid.get(dt);
  vec3_t pos;
  pos.x() = swingX.get(dt);
  pos.y() = swingY.get(dt);
  pos.z() = swingZ.get(dt);
  return pos;
}

} // namespace Capt