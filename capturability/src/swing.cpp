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

void Swing::set(vec2_t foot, vec2_t foot_des) {
  this->foot.x()     = foot.x();
  this->foot.y()     = foot.y();
  this->foot.z()     = 0.0;
  this->foot_des.x() = foot_des.x();
  this->foot_des.y() = foot_des.y();
  this->foot_des.z() = 0.0;

  set(this->foot, this->foot_des);
}

void Swing::set(vec3_t foot, vec3_t foot_des) {
  this->foot.x()     = foot.x();
  this->foot.y()     = foot.y();
  this->foot.z()     = foot.z();
  this->foot_des.x() = foot_des.x();
  this->foot_des.y() = foot_des.y();
  this->foot_des.z() = foot_des.z();

  double dist = sqrt( ( foot_des.x() - foot.x() ) * ( foot_des.x() - foot.x() ) +
                      ( foot_des.y() - foot.y() ) * ( foot_des.y() - foot.y() ) );
  step_time = dist / foot_vel + step_time_min;

  cycloid.set(foot, foot_des, step_time);
}

double Swing::getTime() {
  return step_time;
}

vec3_t Swing::getTraj(double dt) {
  return cycloid.get(dt);
}

} // namespace Capt