#include "base.h"

namespace Capt {

float dot(vec2_t v1, vec2_t v2){
  return v1.x() * v2.x() + v1.y() * v2.y();
}

float cross(vec2_t v1, vec2_t v2){
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

vec3_t mirror(vec3_t v){
  v.y() *= -1;
  return v;
}

vec3_t vec2Tovec3(vec2_t vec2){
  return vec3_t(vec2.x(), vec2.y(), 0.0);
}

vec2_t vec3Tovec2(vec3_t vec3){
  return vec2_t(vec3.x(), vec3.y());
}

int round(float value) {
  int integer = (int)value;

  float decimal = value - integer;
  if(decimal > 0.0f) {
    if (decimal >= 0.5f) {
      integer++;
    }
  }else{
    if (decimal <= -0.5f) {
      integer--;
    }
  }

  return integer;
}

} // namespace Capt