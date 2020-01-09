#include "swing.h"

namespace Capt {

Swing::Swing(Model *model) {
  foot     = vec3_t::Zero();
  foot_des = vec3_t::Zero();

  tau = 0.0;

  model->read(&v_max,  "foot_vel_max");
  model->read(&dt_min, "step_time_min");
  model->read(&h,      "step_height");

  swingUp.set(0, h, 0, 0, dt_min / 2);
  swingDown.set(h, 0, 0, 0, dt_min / 2);
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

  double dist = ( foot_des - foot ).norm();
  tau       = max(0, dt_min / 2 - elapsed) + dist / v_max + dt_min / 2;
  tau_swing = dist / v_max;
}

double Swing::getTime() {
  return tau;
}

// dt = time from support foot exchange
vec3_t Swing::getTraj(double dt) {
  // judge phase
  SwingPhase phase;
  if(dt > tau) {
    phase = LAND;
  }else if(dt < dt_min / 2) {
    phase = OFF;
  }else if(dt < dt_min / 2 + tau_swing) {
    phase = SWING;
  }else{
    phase = ON;
  }

  // calculate desired swing foot position
  vec3_t pos;
  switch (phase) {
  case OFF:
    pos.x() = foot.x();
    pos.y() = foot.y();
    pos.z() = swingUp.get(dt);
    break;
  case SWING:
    pos.x() = ( foot_des.x() - foot.x() ) * ( dt - dt_min / 2 ) / tau_swing + foot.x();
    pos.y() = ( foot_des.y() - foot.y() ) * ( dt - dt_min / 2 ) / tau_swing + foot.y();
    pos.z() = h;
    break;
  case ON:
    pos.x() = foot_des.x();
    pos.y() = foot_des.y();
    pos.z() = swingDown.get(dt - tau_swing - dt_min / 2);
    break;
  case LAND:
    pos.x() = foot_des.x();
    pos.y() = foot_des.y();
    pos.z() = 0;
    break;
  }

  return pos;
}

} // namespace Capt