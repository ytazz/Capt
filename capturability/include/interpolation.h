#ifndef __INTERPOLATION_H__
#define __INTERPOLATION_H__

namespace Capt {

class Interpolation {
public:
  Interpolation();
  ~Interpolation();

  void set(const float xi_0, const float xi_f, const float d_xi_0,
           const float d_xi_f, const float t_f);
  float get(const float t);

private:
  float coef[4];
};
}
#endif