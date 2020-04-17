#include "monitor.h"

#include <limits>

namespace Capt {

Monitor::Monitor(Capturability *capturability) : cap(capturability){

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

  int next_swf_id = cap->grid->xyz.toIndex(cap->grid->xyz.round(stnext.swf));
  int next_icp_id = cap->grid->xy .toIndex(cap->grid->xy .round(stnext.icp));
  printf("next state id: %d,%d\n", next_swf_id, next_icp_id);

  bool next_ok;
  bool cop_ok;

  // check if next state is in capture basin
  int nstep = -1;
  if(cap->isCapturable(next_swf_id, next_icp_id, nstep)){
    printf("next state is %d-step capturable\n", nstep);
    next_ok = true;
  }
  else{
    printf("next state is not capturable\n");
    next_ok = false;
  }

  // calculate cop
  Input in  = cap->calcInput(st, stnext);
  // check if cop is inside support region
  printf("cop(local): %f,%f\n", in.cop.x(), in.cop.y());
  if( cap->isInsideSupport(in.cop) ){
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
  CaptureState cs;
  if(!cap->findNearest(st, stnext, cs)){
    printf("no capturable state found\n");
    return Status::FAIL;
  }
  printf("modified next state: %d,%d  %d-step capturable transition\n", cs.swf_id, cs.icp_id, cs.nstep);

  State stmod;
  stmod.swf = cap->grid->xyz[cs.swf_id];
  stmod.icp = cap->grid->xy [cs.icp_id];
  in = cap->calcInput(st, stmod);
  input.land = state.suf  + vec3_t(in.swf.x(), sign*in.swf.y(), 0.0f);
  input.cop  = state.suf  + vec3_t(in.cop.x(), sign*in.cop.y(), 0.0f);
  input.icp  = input.land + vec3_t(stmod.icp.x(), sign_next*stmod.icp.y(), 0.0f);

  return Status::MODIFIED;
}

} // namespace Capt