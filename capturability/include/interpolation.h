#ifndef __INTERPOLATION_H__
#define __INTERPOLATION_H__

namespace Capt {

class Interpolation {
public:
  Interpolation();
  ~Interpolation();

  void set(const double xi_0, const double xi_f, const double d_xi_0,
           const double d_xi_f, const double t_f);
  double get(const double t);

private:
  double coef[4];
};
}
#endif