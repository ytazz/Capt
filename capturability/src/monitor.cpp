#include "monitor.h"

namespace Capt {

Monitor::Monitor(Model *model, Grid *grid, Capturability *capturability) :
  grid(grid), capturability(capturability){
  swing = new Swing(model);

  model->read(&dt_min, "step_time_min");
}

Monitor::~Monitor(){
}

bool Monitor::check(EnhancedState state, Footstep footstep){
  this->state = state;

  vec2_t rfoot = vec3Tovec2(state.rfoot);
  vec2_t lfoot = vec3Tovec2(state.lfoot);
  vec2_t icp   = vec3Tovec2(state.icp);

  // calculate next landing position
  vec2_t nextLandingPos;
  vec3_t suf;
  if(state.s_suf == FOOT_R) {
    suf = state.rfoot;
  }else{
    suf = state.lfoot;
  }

  double distMin       = 100; // set very large value as initial value
  int    currentFootId = 0;
  for(size_t i = 0; i < state.footstep.size(); i++) {
    if(state.footstep[i].suf == state.s_suf) {
      double dist = ( state.footstep[i].pos - suf ).norm();
      if(dist < distMin) {
        distMin       = dist;
        currentFootId = (int)i;
      }
    }
  }
  nextLandingPos = vec3Tovec2(state.footstep[currentFootId + 1].pos);
  printf("%lf, %lf\n", nextLandingPos.x(), nextLandingPos.y() );

  // calculate current state
  State s_state;
  if(state.s_suf == FOOT_R) {
    s_state.icp.x() = +( icp.x() - rfoot.x() );
    s_state.icp.y() = +( icp.y() - rfoot.y() );
    s_state.swf.x() = +( lfoot.x() - rfoot.x() );
    s_state.swf.y() = +( lfoot.y() - rfoot.y() );
  }else{
    s_state.icp.x() = +( icp.x() - lfoot.x() );
    s_state.icp.y() = -( icp.y() - lfoot.y() );
    s_state.swf.x() = +( rfoot.x() - lfoot.x() );
    s_state.swf.y() = -( rfoot.y() - lfoot.y() );
  }
  if(state.elapsed < dt_min / 2) {
    s_state.elp = state.elapsed;
  }else{
    s_state.elp = dt_min / 2;
  }

  // calculate current state id
  int state_id = grid->roundState(s_state).id;

  // get 1-step capture region (support foot coordinate)
  std::vector<CaptureSet*> region = capturability->getCaptureRegion(state_id, 1);

  // shift capture region (world frame coordinate)
  arr2_t captureRegion;
  captureRegion.resize(region.size() );
  for (size_t i = 0; i < region.size(); i++) {
    vec2_t point = grid->getInput(region[i]->input_id).swf;
    if(state.s_suf == FOOT_R) {
      captureRegion[i] = rfoot + point;
    }else{
      captureRegion[i].x() = lfoot.x() + point.x();
      captureRegion[i].y() = lfoot.y() + point.y();
    }
  }

  // judge 1-step capturable or not
  bool isOneStepCapturable = false;
  min   = ( nextLandingPos - captureRegion[0] ).norm();
  minId = 0;
  for(size_t i = 0; i < region.size(); i++) {
    double dist = ( nextLandingPos - captureRegion[i] ).norm();
    if(dist < min) {
      min                 = dist;
      minId               = (int)i;
      isOneStepCapturable = true;
    }
  }

  // set swing foot trajectory
  if(state.s_suf == FOOT_R) {
    swing->set(lfoot, nextLandingPos, s_state.elp);
  }else{
    swing->set(rfoot, nextLandingPos, s_state.elp);
  }

  // calculate input
  input.duration = swing->getTime();
  input.cop      = vec2Tovec3(grid->getInput(region[minId]->input_id).cop);
  input.icp      = state.icp;
  input.land     = vec2Tovec3(nextLandingPos);

  if(state.s_suf == FOOT_R) {
    input.suf = state.rfoot;
    input.swf = state.lfoot;
  }else{
    input.suf = state.lfoot;
    input.swf = state.rfoot;
  }

  // if 1-step capturable, return true
  return isOneStepCapturable;
}

EnhancedInput Monitor::get(){
  return input;
}

} // namespace Capt