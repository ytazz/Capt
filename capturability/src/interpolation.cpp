#include "interpolation.h"

namespace CA {

Interpolation::Interpolation() {}

Interpolation::~Interpolation() {}

void Interpolation::set(const float xi_0, const float xi_f, const float d_xi_0,
                        const float d_xi_f, const float t_f) {
  coef[0] = xi_0;
  coef[1] = d_xi_0;
  coef[2] =
      (1 / (t_f * t_f)) * (3 * (xi_f - xi_0) - (2 * d_xi_0 + d_xi_f) * t_f);
  coef[3] =
      (1 / (t_f * t_f * t_f)) * (-2 * (xi_f - xi_0) + (d_xi_0 + d_xi_f) * t_f);
}

float Interpolation::get(const float t) {
  return coef[0] + coef[1] * t + coef[2] * t * t + coef[3] * t * t * t;
}
}