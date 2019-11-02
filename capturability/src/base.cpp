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

int round(double value) {
  int result = (int)value;

  double decimal = value - (int)value;
  if (decimal >= 0.5) {
    result += 1;
  }

  return result;
}

} // namespace Capt