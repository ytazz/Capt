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
    double elapsed_ = std::chrono::duration_cast<std::chrono::milliseconds>(end_[i] - start_).count();
    printf("%lf [ms]\n", elapsed_);
  }
}

} // namespace Capt