#ifndef __NODE_H__
#define __NODE_H__

#include "base.h"
#include <iostream>
#include <string>

namespace Capt {

struct Node {
  Node(){
  }

  Node * parent;
  double cost;
  int    step;
};

} // namespace Capt

#endif // __NODE_H__