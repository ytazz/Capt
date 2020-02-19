#ifndef __TIMER_H__
#define __TIMER_H__

#include <iostream>
#include <chrono>
#include <vector>

namespace Capt {

class Timer {
public:
  Timer();
  ~Timer();

  void start();
  void end();

  void print();

  double get(); // [milli sec.]

private:
  std::chrono::system_clock::time_point start_;
  std::chrono::system_clock::time_point end_;
  int                                   elapsed_;
};

} // namespace Capt

#endif // __TIMER_H__