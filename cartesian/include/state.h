#ifndef __STATE_H__
#define __STATE_H__

#include "base.h"
#include <iostream>
#include <string>

namespace Capt {

struct State {
  vec2_t icp;
  vec2_t swf;
  double elp; // elapsed time

  State(){
  }
  State(double icp_x, double icp_y, double swf_x, double swf_y, double elp){
    set(icp_x, icp_y, swf_x, swf_y, elp);
  }
  State(vec2_t icp, vec2_t swf, double elp){
    set(icp, swf, elp);
  }

  void set(double icp_x, double icp_y, double swf_x, double swf_y, double elp){
    this->icp.x() = icp_x;
    this->icp.y() = icp_y;
    this->swf.x() = swf_x;
    this->swf.y() = swf_y;
    this->elp     = elp;
  }

  void set(vec2_t icp, vec2_t swf, double elp){
    this->icp = icp;
    this->swf = swf;
    this->elp = elp;
  }

  void print() {
    printf("icp = [ %lf, %lf ]\n", icp.x(), icp.y() );
    printf("swf = [ %lf, %lf ]\n", swf.x(), swf.y() );
    printf("elp = %1.3lf\n", elp );
  }

  void operator=(const State &state) {
    this->icp = state.icp;
    this->swf = state.swf;
    this->elp = state.elp;
  }
};

} // namespace Capt

#endif // __STATE_H__