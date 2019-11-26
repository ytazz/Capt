#include "trajectory.h"

namespace Capt {

Trajectory::Trajectory(Model *model) : pendulum(model), swing_foot(model){
  model->read(&h, "com_height");
}

Trajectory::~Trajectory(){
}

void Trajectory::set(State state, Input input, vec3_t world_p_com, vec3_t world_p_suf, Foot suf){
  this->suf         = suf;
  this->world_p_suf = world_p_suf;

  vec3_t world_p_cop;
  vec3_t world_p_icp;
  vec3_t world_p_swf0, world_p_swf1;
  world_p_cop.x()  = world_p_suf.x() + input.cop.x();
  world_p_cop.z()  = 0.0;
  world_p_icp.x()  = state.icp.x();
  world_p_icp.z()  = 0.0;
  world_p_swf0.x() = world_p_suf.x() + state.swf.x();
  world_p_swf0.z() = 0.0;
  world_p_swf1.x() = world_p_suf.x() + input.swf.x();
  world_p_swf1.z() = 0.0;
  if(suf == FOOT_R) {
    world_p_cop.y()  = world_p_suf.y() + input.cop.y();
    world_p_icp.y()  = world_p_suf.y() + input.cop.y();
    world_p_swf0.y() = world_p_suf.y() + state.swf.y();
    world_p_swf1.y() = world_p_suf.y() + input.swf.y();
  }else{
    world_p_cop.y()  = world_p_suf.y() - input.cop.y();
    world_p_icp.y()  = world_p_suf.y() - input.cop.y();
    world_p_swf0.y() = world_p_suf.y() - state.swf.y();
    world_p_swf1.y() = world_p_suf.y() - input.swf.y();
  }

  pendulum.setCop(world_p_cop);
  pendulum.setCom(world_p_com);
  pendulum.setIcp(world_p_icp);
  swing_foot.set(world_p_swf0, world_p_swf1);
}

vec3_t Trajectory::getCom(double elapsed_time){
  vec3_t com;
  com.x() = pendulum.getCom(elapsed_time).x();
  com.y() = pendulum.getCom(elapsed_time).y();
  com.z() = h;
  return com;
}

vec3_t Trajectory::getIcp(double elapsed_time){
  vec3_t icp;
  icp.x() = pendulum.getIcp(elapsed_time).x();
  icp.y() = pendulum.getIcp(elapsed_time).y();
  icp.z() = 0.0;
  return icp;
}

vec3_t Trajectory::getFootR(double elapsed_time){
  if(suf == FOOT_R) {
    return world_p_suf;
  }else{
    return swing_foot.getTraj(elapsed_time);
  }
}

vec3_t Trajectory::getFootL(double elapsed_time){
  if(suf == FOOT_L) {
    return world_p_suf;
  }else{
    return swing_foot.getTraj(elapsed_time);
  }
}

} // namespace Capt