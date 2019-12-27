#include "monitor.h"

namespace Capt {

Monitor::Monitor(Grid *grid, Capturability *capturability){
  this->grid          = grid;
  this->capturability = capturability;
}

Monitor::~Monitor(){
}

bool Monitor::check(EnhancedState state){
  vec2_t rfoot = vec3Tovec2(state.rfoot);
  vec2_t lfoot = vec3Tovec2(state.lfoot);
  vec2_t icp   = vec3Tovec2(state.icp);

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
    s_state.elapsed = state.elapsed;
  }else{
    s_state.elapsed = dt_min / 2;
  }

  // calculate current state id
  int state_id = grid->roundState(s_state).id;

  // get 1-step capture region (support foot coordinate)
  std::vector<CaptureSet*> region = capturability->getCaptureRegion(state_id, 1);

  // shift capture region (world frame coordinate)
  arr2_t captureRegion;
  captureRegion.resize(region.size() );
  for (size_t i = 0; i < region.size(); i++) {
    vec2_t point = grid->getInput(region[i].input_id).swf;
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

  // if 1-step capturable, return true
  return isOneStepCapturable;
}

} // namespace Capt