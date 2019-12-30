#include "trajectory.h"

namespace Capt {

Trajectory::Trajectory(Model *model) : pendulum(model), swing(model){
  model->read(&h, "step_height");
}

Trajectory::~Trajectory(){
}

void Trajectory::set(EnhancedInput input, Foot suf){
  this->input = input;
  this->suf   = suf;

  pendulum.setCop(vec3Tovec2(input.cop) );
  pendulum.setIcp(vec3Tovec2(input.icp) );
  swing.set(input.swf, input.land, 0.0);
}

vec3_t Trajectory::getCop(double elapsed){
  return vec2Tovec3(pendulum.getCop(elapsed) );
}

vec3_t Trajectory::getIcp(double elapsed){
  return vec2Tovec3(pendulum.getIcp(elapsed) );
}

vec3_t Trajectory::getFootR(double elapsed){
  if(suf == FOOT_R) {
    return input.suf;
  }else{
    return swing.getTraj(elapsed);
  }
}

vec3_t Trajectory::getFootL(double elapsed){
  if(suf == FOOT_L) {
    return input.suf;
  }else{
    return swing.getTraj(elapsed);
  }
}

} // namespace Capt