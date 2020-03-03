#include "trajectory.h"

namespace Capt {

Trajectory::Trajectory(Model *model, Param *param) : pendulum(model), swing(model, param){
}

Trajectory::~Trajectory(){
}

void Trajectory::set(EnhancedInput input, Foot suf){
  this->input = input;
  this->suf   = suf;

  pendulum.setCop(vec3Tovec2(input.cop) );
  pendulum.setIcp(vec3Tovec2(input.icp) );

  swing.set(input.swf, input.land);
}

vec3_t Trajectory::getCop(double dt){
  return vec2Tovec3(pendulum.getCop(dt) );
}

vec3_t Trajectory::getIcp(double dt){
  return vec2Tovec3(pendulum.getIcp(dt) );
}

vec3_t Trajectory::getFootR(double dt){
  if(suf == FOOT_R) {
    return input.suf;
  }else{
    return swing.getTraj(dt);
  }
}

vec3_t Trajectory::getFootL(double dt){
  if(suf == FOOT_L) {
    return input.suf;
  }else{
    return swing.getTraj(dt);
  }
}

} // namespace Capt