#include "timer.h"

namespace Capt {

Timer::Timer(){
  end_.clear();

  start_ = std::chrono::system_clock::now();
}

Timer::~Timer(){
}

void Timer::start(){
  start_ = std::chrono::system_clock::now();
}

void Timer::end(){
  end_.push_back(std::chrono::system_clock::now() );
}

void Timer::print(){
  for(size_t i = 0; i < end_.size(); i++) {
    int elapsed_ = std::chrono::duration_cast<std::chrono::microseconds>(end_[i] - start_).count();
    printf("%d.%d [ms]\n", elapsed_ / 1000, elapsed_ % 1000);
  }
}

} // namespace Capt