#include "swing.h"

namespace Capt {

Swing::Swing(Model *model, Param *param) {
  foot     = vec3_t::Zero();
  foot_des = vec3_t::Zero();

  tau = 0.0;

  model->read(&v_max, "foot_vel_max");
  param->read(&z_max, "swf_z_max");

  // calculate minimum swing up/down time
  dt_min = z_max / v_max;
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

  dist_x =  foot_des.x() - foot.x();
  dist_y =  foot_des.y() - foot.y();
  dist   = sqrt( dist_x * dist_x + dist_y * dist_y );

  tau = ( 2 * z_max + dist ) / v_max;
  // tau = ( 2 * z_max - foot.z() + dist ) / v_max;
}

double Swing::getTime() {
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
    pos.x() = foot_des.x();
    pos.y() = foot_des.y();
    pos.z() = v_max * ( tau - dt );
  }else{                        // landing
    pos.x() = foot_des.x();
    pos.y() = foot_des.y();
    pos.z() = foot_des.z();
  }

  return pos;
}

} // namespace Capt