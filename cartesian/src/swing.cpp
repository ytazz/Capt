#include "swing.h"

namespace Capt {

Swing::Swing(Model *model, Param *param) {
  foot = vec3_t::Zero();
  land = vec3_t::Zero();

  tau = 0.0;

  model->read(&v_max, "foot_vel_max");
  param->read(&z_max, "swf_z_max");

  // calculate minimum swing up/down time
  dt_min = z_max / v_max;
}

Swing::~Swing() {
}

void Swing::set(vec2_t foot, vec2_t land) {
  this->foot.x() = foot.x();
  this->foot.y() = foot.y();
  this->foot.z() = 0.0;
  this->land.x() = land.x();
  this->land.y() = land.y();
  this->land.z() = 0.0;

  set(this->foot, this->land);
}

void Swing::set(vec3_t foot, vec3_t land) {
  this->foot.x() = foot.x();
  this->foot.y() = foot.y();
  this->foot.z() = foot.z();
  this->land.x() = land.x();
  this->land.y() = land.y();
  this->land.z() = land.z();

  dist_x =  land.x() - foot.x();
  dist_y =  land.y() - foot.y();
  dist   = sqrt( dist_x * dist_x + dist_y * dist_y );

  tau = ( 2 * z_max - foot.z() + dist ) / v_max;
}

double Swing::getDuration() {
  return tau;
}

vec3_t Swing::getTraj(double dt) {
  vec3_t pos;

  // phase
  if(dt < dt_min) { // swing up
    pos.x() = foot.x();
    pos.y() = foot.y();
    pos.z() = v_max * dt;
  }else if(dt < tau - dt_min) { // swing
    pos.x() = foot.x() + v_max * ( dist_x / dist ) * ( dt - dt_min );
    pos.y() = foot.y() + v_max * ( dist_y / dist ) * ( dt - dt_min );
    pos.z() = z_max;
  }else if(dt < tau) {          // swing down
    pos.x() = land.x();
    pos.y() = land.y();
    pos.z() = v_max * ( tau - dt );
  }else{                        // landing
    pos.x() = land.x();
    pos.y() = land.y();
    pos.z() = land.z();
  }

  return pos;
}

} // namespace Capt