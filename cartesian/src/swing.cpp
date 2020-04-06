#include "swing.h"

namespace Capt {

Swing::Swing(Model *model, Param *param) {
  foot = vec3_t::Zero();
  land = vec3_t::Zero();

  tau        = 0.0;
  tau_offset = 0.0;

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
  // tau        = 0.3;
  // v_max      = ( 2 * z_max + dist ) / tau;
  tau_offset = foot.z() / v_max;
  dt_min     = z_max / v_max;
}

float Swing::getDuration() {
  return tau;
}

vec3_t Swing::getTraj(float dt) {
  // elapsed time from support foot exchange
  float elapsed = tau_offset + dt;
  // remaining time until the foot lands on the ground
  float remained = tau - dt;

  // calculate swing foot position at current phase
  vec3_t pos;
  if(elapsed <= dt_min) {
    // swing up
    pos.x() = foot.x();
    pos.y() = foot.y();
    pos.z() = v_max * elapsed;
  }else if(remained >= dt_min) {
    // swing
    pos.x() = foot.x() + v_max * ( dist_x / dist ) * ( elapsed - dt_min );
    pos.y() = foot.y() + v_max * ( dist_y / dist ) * ( elapsed - dt_min );
    pos.z() = z_max;
  }else if(remained > 0) {
    // swing down
    pos.x() = land.x();
    pos.y() = land.y();
    pos.z() = v_max * remained;
  }else{
    // landing
    pos.x() = land.x();
    pos.y() = land.y();
    pos.z() = land.z();
  }

  return pos;
}

bool Swing::isSwingDown(float dt){
  bool flag = false;

  // remaining time until the foot lands on the ground
  float remained = tau - dt;

  // judge swing down phase or not
  if(remained <= dt_min) {
    flag = true;
  }

  return flag;
}

} // namespace Capt