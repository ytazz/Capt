#ifndef __STATE_H__
#define __STATE_H__

#include "base.h"
#include <iostream>
#include <string>

namespace Capt {

struct State {
  vec2_t icp;
  vec3_t swf;

  State(){
  }
  State(float icp_x, float icp_y, float swf_x, float swf_y, float swf_z){
    set(icp_x, icp_y, swf_x, swf_y, swf_z);
  }
  State(vec2_t icp, vec3_t swf){
    set(icp, swf);
  }

  void set(float icp_x, float icp_y, float swf_x, float swf_y, float swf_z){
    this->icp.x() = icp_x;
    this->icp.y() = icp_y;
    this->swf.x() = swf_x;
    this->swf.y() = swf_y;
    this->swf.z() = swf_z;
  }

  void set(vec2_t icp, vec3_t swf){
    this->icp = icp;
    this->swf = swf;
  }

  void print() {
    printf("icp = [ %lf, %lf ]\n", icp.x(), icp.y() );
    printf("swf = [ %lf, %lf, %lf ]\n", swf.x(), swf.y(), swf.z() );
  }

  void operator=(const State &state) {
    this->icp = state.icp;
    this->swf = state.swf;
  }
};

} // namespace Capt

#endif // __STATE_H__