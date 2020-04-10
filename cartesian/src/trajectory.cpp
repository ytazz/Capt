#include "trajectory.h"

namespace Capt {

Trajectory::Trajectory(Pendulum* pendulum, Swing* swing) : pendulum(pendulum), swing(swing){
}

Trajectory::~Trajectory(){
}

void Trajectory::set(const EnhancedState& state, const EnhancedInput& input){
  this->state = state;
  this->input = input;

  pendulum->setIcp(vec3Tovec2(state.icp));
  pendulum->setCop(vec3Tovec2(input.cop));

  swing->set(state.swf, input.land);
}

vec3_t Trajectory::getCop(float t){
  return vec2Tovec3(pendulum->getCop(t) );
}

vec3_t Trajectory::getIcp(float t){
  return vec2Tovec3(pendulum->getIcp(t) );
}

vec3_t Trajectory::getFootR(float t){
  return (state.s_suf == FOOT_R ? state.suf : swing->getTraj(t));
}

vec3_t Trajectory::getFootL(float t){
  return (state.s_suf == FOOT_L ? state.suf : swing->getTraj(t));
}

} // namespace Capt