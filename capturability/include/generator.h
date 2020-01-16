#ifndef __GENERATOR_H__
#define __GENERATOR_H__

#include <iostream>
#include <vector>
#include "base.h"
#include "swing.h"

namespace Capt {

class Generator {
public:
  Generator(Model *model);
  ~Generator();

  void calc(Footstep *footstep);

private:
  Swing *swing;

  double omega;
};

} // namespace Capt

#endif // __GENERATOR_H__