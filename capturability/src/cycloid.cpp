#include "cycloid.h"

namespace Capt {

Cycloid::Cycloid(){
}

Cycloid::~Cycloid(){
}

void Cycloid::set(vec2_t p0, vec2_t pf, double t){
  double dist = (pf-p0).norm();

  direction = (pf-p0).th;
  radius    = dist/(2*M_PI);
  T         = t;
  omega     = M_PI/T;
}

vec3_t Cycloid::get(double t){
  double theta = 0.0;
  if(t<0) {
    theta = 0.0;
  }else if(t>T) {
    theta = omega*T;
  }else{
    theta = omega*t;
  }

  vec3_t pos;
  pos.x() = radius*(theta-sin(theta))*cos(direction);
  pos.y() = radius*(theta-sin(theta))*sin(direction);
  pos.z() = radius*(1-cos(theta));

  return pos;
}

} // namespace Capt