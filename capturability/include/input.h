#ifndef __INPUT_H__
#define __INPUT_H__

#include "base.h"
#include <iostream>
#include <string>

namespace Capt {

struct Input {
  vec2_t swf;

  Input(){
  }
  Input(double swf_x, double swf_y){
    set(swf_x, swf_y);
  }
  Input(vec2_t swf){
    set(swf);
  }

  void set(double swf_x, double swf_y){
    this->swf.x() = swf_x;
    this->swf.y() = swf_y;
  }

  void set(vec2_t swf){
    this->swf = swf;
  }

  void print() {
    printf("swf = [ %lf, %lf ]\n", swf.x(), swf.y() );
  }

  void operator=(const Input &input) {
    this->swf = input.swf;
  }
};

} // namespace Capt

#endif // __INPUT_H__