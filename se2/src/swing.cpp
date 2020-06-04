#include "swing.h"

namespace Capt {

Swing::Swing(Model *model) {
  foot = vec3_t::Zero();
  land = vec3_t::Zero();

  tau_ascend  = 0.0f;
  tau_travel  = 0.0f;
  tau_descend = 0.0f;

  model->read(&v_max, "foot_vel_max");
  model->read(&z_max, "swing_height_max");
}

Swing::~Swing() {
}

void Swing::set(vec3_t foot, vec3_t land) {
  this->foot = foot;
  this->land = land;

  dist_x =  land.x() - foot.x();
  dist_y =  land.y() - foot.y();
  dist   = sqrt( dist_x * dist_x + dist_y * dist_y );

  tau_ascend  = (z_max - foot.z())/v_max;
  tau_travel  = dist/v_max;
  tau_descend = z_max/v_max;

  // tau        = 0.3;
  // v_max      = ( 2 * z_max + dist ) / tau;
  //tau_offset = foot.z() / v_max;
}

float Swing::getDuration() {
  return tau_ascend + tau_travel + tau_descend;
}

vec3_t Swing::getTraj(float t) {
  // calculate swing foot position at current phase
  vec3_t pos;

  // ascending
  if(0 <= t && t < tau_ascend) {
    pos.x() = foot.x();
    pos.y() = foot.y();
    pos.z() = foot.z() + v_max * t;
  }
  // traveling to landing position
  if(tau_ascend <= t && t < tau_ascend + tau_travel) {
    pos.x() = foot.x() + v_max * (dist_x/dist)*(t - tau_ascend);
    pos.y() = foot.y() + v_max * (dist_y/dist)*(t - tau_ascend);
    pos.z() = z_max;
  }
  // descending
  if(tau_ascend + tau_travel <= t && t < tau_ascend + tau_travel + tau_descend) {
    pos.x() = land.x();
    pos.y() = land.y();
    pos.z() = z_max - v_max*(t - (tau_ascend + tau_travel));
  }
  // after landing
  if(t >= tau_ascend + tau_travel + tau_descend){
    pos.x() = land.x();
    pos.y() = land.y();
    pos.z() = land.z();
  }

  return pos;
}
/*
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
*/

} // namespace Capt