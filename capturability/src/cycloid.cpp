#include "cycloid.h"

namespace Capt {

Cycloid::Cycloid(){
}

Cycloid::~Cycloid(){
}

void Cycloid::set(vec3_t p0, vec3_t pf, double t){
  double dist_x = pf.x() - p0.x();
  double dist_y = pf.y() - p0.y();
  double dist   = sqrt(dist_x * dist_x + dist_y * dist_y);

  origin = p0;

  direction = atan2(dist_y, dist_x);
  radius    = dist / ( 2 * M_PI );
  T         = t;
  omega     = 2 * M_PI / T;
}

vec3_t Cycloid::get(double t){
  double theta = 0.0;

  if(t < 0) {
    theta = 0.0;
  }else if(t > T) {
    theta = omega * T;
  }else{
    theta = omega * t;
  }

  vec3_t pos;
  pos.x() = origin.x() + radius * ( theta - sin(theta) ) * cos(direction);
  pos.y() = origin.y() + radius * ( theta - sin(theta) ) * sin(direction);
  pos.z() = origin.z() + radius * ( 1 - cos(theta) );

  return pos;
}

} // namespace Capt