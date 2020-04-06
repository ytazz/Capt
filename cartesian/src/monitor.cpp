#include "monitor.h"

#include <limits>

namespace Capt {

const float inf = std::numeric_limits<float>::max();

Monitor::Monitor(Model *model, Param *param, Grid *grid, Capturability *capturability) :
  grid(grid), capturability(capturability){

}

Monitor::~Monitor(){
}

Status Monitor::check(EnhancedState state, Footstep footstep){
  // support foot and swing foot position
  vec3_t suf = (state.s_suf == FOOT_R ? state.rfoot : state.lfoot);
  vec3_t swf = (state.s_suf == FOOT_R ? state.lfoot : state.rfoot);
  float sign = (state.s_suf == FOOT_R ? 1.0f : -1.0f);

  // calculate next landing position
  float distMin       = inf; // set very large value as initial value
  int   currentFootId = 0;
  for(size_t i = 0; i < state.footstep.size(); i++) {
    if(state.footstep[i].suf == state.s_suf) {
      float dist = vec3Tovec2(state.footstep[i].pos - suf ).norm();
      if(dist < distMin) {
        distMin       = dist;
        currentFootId = (int)i;
      }
    }
  }
  if(currentFootId == ( (int)state.footstep.size() - 1 ) ) {
    return Status::FINISH;
  }
  vec3_t suf_next  = state.footstep[currentFootId + 1].pos;
  vec3_t swf_next  = suf;
  float  sign_next = -sign;
  vec3_t icp_next  = state.footstep[currentFootId + 1].icp;

  // calculate current state and next state
  State st, stnext;
  st.icp.x() =      ( state.icp.x() - suf.x() );
  st.icp.y() = sign*( state.icp.y() - suf.y() );
  st.swf.x() =      ( swf.x() - suf.x() );
  st.swf.y() = sign*( swf.y() - suf.y() );
  st.swf.z() = swf.z();

  stnext.icp.x() =           ( icp_next.x() - suf_next.x() );
  stnext.icp.y() = sign_next*( icp_next.y() - suf_next.y() );
  stnext.swf.x() =           ( swf_next.x() - suf_next.x() );
  stnext.swf.y() = sign_next*( swf_next.y() - suf_next.y() );
  stnext.swf.z() = swf_next.z();

  // calculate current state id
  st.print();
  stnext.print();

  int state_id = grid->roundState(st);
  int next_id  = grid->roundState(stnext);
  printf("state_id %d,  next_id %d\n", state_id, next_id);

  // get 1-step capture region (support foot coordinate)
  if(capturability->isCapturable(next_id, 1))
    return Status::SUCCESS;

  return Status::FAIL;
}

} // namespace Capt