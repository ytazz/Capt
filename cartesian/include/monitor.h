#ifndef __MONITOR_H__
#define __MONITOR_H__

#include <iostream>
#include <vector>
#include "capturability.h"

namespace Capt {

class Monitor {
public:
  Monitor(Capturability *capturability);
  ~Monitor();

  Status check(const EnhancedState& state, EnhancedInput& input);

private:
  Capturability *cap;
};

} // namespace Capt

#endif // __MONITOR_H__