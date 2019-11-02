#ifndef __STATE_H__
#define __STATE_H__

#include "base.h"
#include <iostream>
#include <string>

namespace Capt {

struct State {
  vec2_t icp;
  vec2_t swf;

  State(){
  }
  State(double icp_x, double icp_y, double swf_x, double swf_y){
    set(icp_x, icp_y, swf_x, swf_y);
  }
  State(vec2_t icp, vec2_t swf){
    set(icp, swf);
  }

  void set(double icp_x, double icp_y, double swf_x, double swf_y){
    this->icp.x() = icp_x;
    this->icp.y() = icp_y;
    this->swf.x() = swf_x;
    this->swf.y() = swf_y;
  }

  void set(vec2_t icp, vec2_t swf){
    this->icp = icp;
    this->swf = swf;
  }

  void print() {
    printf("icp = [ %lf, %lf ]\n", icp.x(), icp.y() );
    printf("swf = [ %lf, %lf ]\n", swf.x(), swf.y() );
  }

  void operator=(const State &state) {
    this->icp = state.icp;
    this->swf = state.swf;
  }
};

} // namespace Capt

#endif // __STATE_H__