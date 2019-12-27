#ifndef __MONITOR_H__
#define __MONITOR_H__

#include <iostream>
#include <chrono>
#include <vector>

namespace Capt {

class Monitor {
public:
  Monitor();
  ~Monitor();

  bool check();

private:
};

} // namespace Capt

#endif // __MONITOR_H__