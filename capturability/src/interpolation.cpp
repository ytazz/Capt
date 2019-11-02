#include "interpolation.h"

namespace Capt {

Interpolation::Interpolation() {}

Interpolation::~Interpolation() {}

void Interpolation::set(const double xi_0, const double xi_f, const double d_xi_0,
                        const double d_xi_f, const double t_f) {
  coef[0] = xi_0;
  coef[1] = d_xi_0;
  coef[2] =
      (1 / (t_f * t_f)) * (3 * (xi_f - xi_0) - (2 * d_xi_0 + d_xi_f) * t_f);
  coef[3] =
      (1 / (t_f * t_f * t_f)) * (-2 * (xi_f - xi_0) + (d_xi_0 + d_xi_f) * t_f);
}

double Interpolation::get(const double t) {
  return coef[0] + coef[1] * t + coef[2] * t * t + coef[3] * t * t * t;
}
}