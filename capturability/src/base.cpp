#include "base.h"

namespace Capt {

double dot(vec2_t v1, vec2_t v2){
  return v1.x() * v2.x() + v1.y() * v2.y();
}

double cross(vec2_t v1, vec2_t v2){
  return v1.x() * v2.y() - v1.y() * v2.x();
}

vec2_t normal(vec2_t v) {
  // rotate -90 deg around +z direction
  vec2_t normal_vector(v.y(), -v.x() );
  return normal_vector;
}

vec2_t mirror(vec2_t v){
  v.y() *= -1;
  return v;
}

vec3_t vec2Tovec3(vec2_t vec2){
  vec3_t vec3;
  vec3.x() = vec2.x();
  vec3.y() = vec2.y();
  vec3.z() = 0.0;
  return vec3;
}

vec2_t vec3Tovec2(vec3_t vec3){
  vec2_t vec2;
  vec2.x() = vec3.x();
  vec2.y() = vec3.y();
  return vec2;
}

int round(double value) {
  int integer = (int)value;

  double decimal = value - integer;
  if(decimal > 0) {
    if (decimal >= 0.5) {
      integer += 1;
    }
  }else{
    if (decimal <= -0.5) {
      integer -= 1;
    }
  }

  return integer;
}

} // namespace Capt