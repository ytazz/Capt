#ifndef __CYCLOID_H__
#define __CYCLOID_H__

#include "loader.h"
#include <string>
#include <math.h>

namespace Capt {

class Cycloid {
public:
  Cycloid();
  ~Cycloid();

  // p0: initial position
  // pf: terminal position
  // t : stepping time
  void set(vec3_t p0, vec3_t pf, double t);
  // t : elapsed time
  vec3_t get(double t);

private:
  double height;
  double direction;
  double radius;
  double omega;
  double T; // stepping time
};

} // namespace Capt

#endif // __CYCLOID_H__