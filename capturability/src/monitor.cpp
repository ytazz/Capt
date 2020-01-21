#include "monitor.h"

namespace Capt {

Monitor::Monitor(Model *model, Grid *grid, Capturability *capturability) :
  grid(grid), capturability(capturability){
  swing    = new Swing(model);
  pendulum = new Pendulum(model);

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
  vec2_t suf;
  if(state.s_suf == FOOT_R) {
    suf = rfoot;
  }else{
    suf = lfoot;
  }

  double distMin       = 100; // set very large value as initial value
  int    currentFootId = 0;
  for(size_t i = 0; i < state.footstep.size(); i++) {
    if(state.footstep[i].suf == state.s_suf) {
      double dist = ( vec3Tovec2(state.footstep[i].pos) - suf ).norm();
      if(dist < distMin) {
        distMin       = dist;
        currentFootId = (int)i;
      }
    }
  }
  nextLandingPos = vec3Tovec2(state.footstep[currentFootId + 1].pos);

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
  s_state.print();
  int state_id = grid->roundState(s_state).id;
  printf("state_id %d\n", state_id);

  // get 1-step capture region (support foot coordinate)
  std::vector<CaptureSet*> region = capturability->getCaptureRegion(state_id, 1);

  // shift capture region (world frame coordinate)
  captureRegion.clear();
  captureRegion.resize(region.size() );
  for (size_t i = 0; i < region.size(); i++) {
    vec2_t point = grid->getInput(region[i]->input_id).swf;
    if(state.s_suf == FOOT_R) {
      captureRegion[i] = rfoot + point;
    }else{
      captureRegion[i].x() = lfoot.x() + point.x();
      captureRegion[i].y() = lfoot.y() - point.y();
    }
  }

  // judge 1-step capturable or not
  bool isOneStepCapturable = false;
  min = 100;
  for(size_t i = 0; i < region.size(); i++) {
    double dist = ( nextLandingPos - captureRegion[i] ).norm();
    if(dist < min) {
      min = dist;
    }
  }
  printf("min %lf\n", min);
  if(min < 0.05) {
    isOneStepCapturable = true;
  }

  // set swing foot trajectory
  if(state.s_suf == FOOT_R) {
    swing->set(lfoot, nextLandingPos, s_state.elp);
  }else{
    swing->set(rfoot, nextLandingPos, s_state.elp);
  }

  // calculate landing position (support foot coord.)
  vec2_t d_swf;
  if(state.s_suf == FOOT_R) {
    d_swf = nextLandingPos - rfoot;
  }else{
    d_swf      = nextLandingPos - lfoot;
    d_swf.y() *= -1;
  }
  // printf("nextLandingPos %1.2lf, %1.2lf\n", nextLandingPos.x(), nextLandingPos.y() );
  // printf("rfoot          %1.2lf, %1.2lf\n", rfoot.x(), rfoot.y() );
  // printf("lfoot          %1.2lf, %1.2lf\n", lfoot.x(), lfoot.y() );
  // printf("d_swf          %1.2lf, %1.2lf\n", d_swf.x(), d_swf.y() );

  // calculate landing foot index
  int swfId = grid->indexSwf(d_swf);
  printf("swf   %1.2lf, %1.2lf\n", d_swf.x(), d_swf.y() );
  printf("swfId %d\n", swfId);

  // select best icp_hat position & step duration
  vec2_t icp_hat, icp_hat_;
  vec2_t cop     = vec2_t(0, 0);
  vec2_t cop_    = vec2_t(0, 0);
  vec2_t icp_    = vec2_t(0, 0);
  vec2_t icp_des = vec3Tovec2(state.footstep[currentFootId + 1].icp);
  min = 100;
  for (size_t i = 0; i < region.size(); i++) {
    Input input = grid->getInput(region[i]->input_id);
    if(grid->indexSwf(input.swf) == swfId) {
      if(state.s_suf == FOOT_R) {
        icp_.x() =  s_state.icp.x() + suf.x();
        icp_.y() =  s_state.icp.y() + suf.y();
        cop_.x() =  input.cop.x() + suf.x();
        cop_.y() =  input.cop.y() + suf.y();
      }else{
        icp_.x() =  s_state.icp.x() + suf.x();
        icp_.y() = -s_state.icp.y() + suf.y();
        cop_.x() =  input.cop.x() + suf.x();
        cop_.y() = -input.cop.y() + suf.y();
      }
      pendulum->setIcp(icp_);
      pendulum->setCop(cop_);
      icp_hat_ = pendulum->getIcp(swing->getTime() );

      double dist = ( icp_hat_ - icp_des ).norm();
      printf("dist %1.3lf cop %1.3lf, %1.3lf\n", dist, cop_.x(), cop_.y() );
      printf("icp_hat  %1.3lf, %1.3lf\n", icp_hat_.x(), icp_hat_.y() );
      if(dist < min) {
        cop     = cop_;
        icp_hat = icp_hat_;
        min     = dist;
      }
    }
  }

  // calculate best cop
  printf("icp_des  %1.3lf, %1.3lf\n", icp_des.x(), icp_des.y() );
  printf("icp_hat  %1.3lf, %1.3lf\n", icp_hat.x(), icp_hat.y() );
  printf("duration %1.3lf\n", swing->getTime() );
  printf("cop      %1.3lf, %1.3lf\n", cop.x(), cop.y() );

  // calculate input
  if(isOneStepCapturable) {
    input.elapsed  = s_state.elp;
    input.duration = swing->getTime();
    input.land     = vec2Tovec3(nextLandingPos);

    if(state.s_suf == FOOT_R) {
      input.suf = state.rfoot;
      input.swf = state.lfoot;
      input.cop = vec2Tovec3(cop);
      input.icp = vec2Tovec3(s_state.icp) + state.rfoot;
    }else{
      input.suf = state.lfoot;
      input.swf = state.rfoot;
      input.cop = vec2Tovec3(cop);
      input.icp = vec2Tovec3(mirror(s_state.icp) ) + state.lfoot;
    }
  }

  // if 1-step capturable, return true
  return isOneStepCapturable;
}

EnhancedInput Monitor::get(){
  return input;
}

std::vector<CaptData> Monitor::getCaptureRegion(){
  std::vector<CaptData> region;

  region.resize(captureRegion.size() );
  for(size_t i = 0; i < captureRegion.size(); i++) {
    region[i].pos   = vec2Tovec3(captureRegion[i]);
    region[i].nstep = 1;
  }

  return region;
}

} // namespace Capt