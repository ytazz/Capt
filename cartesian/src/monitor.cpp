#include "monitor.h"

#include <limits>

namespace Capt {

Monitor::Monitor(Capturability *capturability) : capturability(capturability){

}

Monitor::~Monitor(){
}

Status Monitor::check(const EnhancedState& state, EnhancedInput& input){
  float sign = (state.s_suf == FOOT_R ? 1.0f : -1.0f);

  vec3_t suf_next  = input.land;
  vec3_t swf_next  = state.suf;
  float  sign_next = -sign;
  vec3_t icp_next  = input.icp;

  // calculate current state and next state
  State st, stnext;
  st.icp.x() =      (state.icp.x() - state.suf.x());
  st.icp.y() = sign*(state.icp.y() - state.suf.y());
  st.swf.x() =      (state.swf.x() - state.suf.x());
  st.swf.y() = sign*(state.swf.y() - state.suf.y());
  st.swf.z() = state.swf.z();

  stnext.icp.x() =           (icp_next.x() - suf_next.x());
  stnext.icp.y() = sign_next*(icp_next.y() - suf_next.y());
  stnext.swf.x() =           (swf_next.x() - suf_next.x());
  stnext.swf.y() = sign_next*(swf_next.y() - suf_next.y());
  stnext.swf.z() = swf_next.z();

  int swf_id      = capturability->grid->roundSwf(st.swf);
  int icp_x_id    = capturability->grid->icp_x.round(st.icp.x());
  int icp_y_id    = capturability->grid->icp_y.round(st.icp.y());
  int next_swf_id = capturability->grid->roundSwf(stnext.swf);
  int next_icp_id = capturability->grid->roundIcp(stnext.icp);
  printf("next state id: %d,%d\n", next_swf_id, next_icp_id);

  bool next_ok;
  bool cop_ok;

  // check if next state is in capture basin
  int nstep = -1;
  if(capturability->isCapturable(next_swf_id, next_icp_id, nstep)){
    printf("next state is %d-step capturable\n", nstep);
    next_ok = true;
  }
  else{
    printf("next state is not capturable\n");
    next_ok = false;
  }

  // calculate cop
  Input in  = capturability->calcInput(st, stnext);
  // check if cop is inside support region
  if( capturability->isInsideSupport(in.cop) ){
    printf("cop is inside support\n");
    cop_ok = true;
  }
  else{
    printf("cop is outside support\n");
    cop_ok = false;
  }

  if(next_ok && cop_ok){
    input.cop = state.suf + vec3_t(in.cop.x(), sign*in.cop.y(), 0.0f);
    return Status::SUCCESS;
  }

  // find modified next state that can be transitioned from current state and is capturable
  int mod_swf_id;
  int mod_icp_id;
  capturability->findNearest(swf_id, icp_x_id, icp_y_id, next_swf_id, next_icp_id, mod_swf_id, mod_icp_id);
  if(mod_swf_id == -1){
    printf("no capturable state found\n");
    return Status::FAIL;
  }
  printf("modified next state: %d,%d\n", mod_swf_id, mod_icp_id);

  State stmod;
  stmod.swf = capturability->grid->swf[mod_swf_id];
  stmod.icp = capturability->grid->icp[mod_icp_id];
  in = capturability->calcInput(st, stmod);
  input.land = state.suf  + vec3_t(in.swf.x(), sign*in.swf.y(), 0.0f);
  input.cop  = state.suf  + vec3_t(in.cop.x(), sign*in.cop.y(), 0.0f);
  input.icp  = input.land + vec3_t(stmod.icp.x(), sign_next*stmod.icp.y(), 0.0f);

  return Status::MODIFIED;
}

} // namespace Capt