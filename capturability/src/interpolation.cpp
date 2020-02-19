#include "interpolation.h"

namespace Capt {

Interp3::Interp3() {
}

Interp3::~Interp3() {
}

void Interp3::set(const double x0, const double xf, const double v0,
                  const double vf, const double tf) {
  coef[0] = x0;
  coef[1] = v0;
  coef[2] =
    ( 1 / ( tf * tf ) ) * ( 3 * ( xf - x0 ) - ( 2 * v0 + vf ) * tf );
  coef[3] =
    ( 1 / ( tf * tf * tf ) ) * ( -2 * ( xf - x0 ) + ( v0 + vf ) * tf );
}

double Interp3::get(const double t) {
  return coef[0] + coef[1] * t + coef[2] * t * t + coef[3] * t * t * t;
}

Interp2::Interp2() {
}

Interp2::~Interp2() {
}

void Interp2::set(const double x0, const double xf, const double vf, const double tf) {
  coef[0] = x0;
  coef[1] = ( 2 * ( xf - x0 ) - vf * tf ) / tf;
  coef[2] = ( x0 - xf + vf * tf ) / ( tf * tf );
}

double Interp2::get(const double t) {
  return coef[0] + coef[1] * t + coef[2] * t * t;
}

}