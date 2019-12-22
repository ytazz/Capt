#include "trajectory.h"

namespace Capt {

Trajectory::Trajectory(Model *model) : pendulum(model), swing_foot(model){
  model->read(&h, "step_height");
}

Trajectory::~Trajectory(){
}

void Trajectory::set(planner::Output input, Foot suf){
  this->input = input;
  this->suf   = suf;

  pendulum.setCop(vec3Tovec2(input.cop) );
  pendulum.setIcp(vec3Tovec2(input.icp) );
  swing_foot.set(input.swf, input.land);

  time = input.alpha * input.duration;
}

vec3_t Trajectory::getCop(double elapsed){
  return vec2Tovec3(pendulum.getCop(time + elapsed) );
}

vec3_t Trajectory::getIcp(double elapsed){
  return vec2Tovec3(pendulum.getIcp(time + elapsed) );
}

vec3_t Trajectory::getFootR(double elapsed){
  if(suf == FOOT_R) {
    return input.suf;
  }else{
    return swing_foot.getTraj(time + elapsed);
  }
}

vec3_t Trajectory::getFootL(double elapsed){
  if(suf == FOOT_L) {
    return input.suf;
  }else{
    return swing_foot.getTraj(time + elapsed);
  }
}

} // namespace Capt