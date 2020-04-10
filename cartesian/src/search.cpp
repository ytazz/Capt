#include "search.h"

namespace Capt {

#if 0
Search::Search(Grid *grid, Tree *tree) :
  grid(grid), tree(tree){
}

Search::~Search() {
}

void Search::clear(){
  seq.clear();
  region.clear();
}

void Search::setStart(vec3_t rfoot, vec3_t lfoot, vec3_t icp, Foot s_suf){
  this->rfoot = vec3Tovec2(rfoot);
  this->lfoot = vec3Tovec2(lfoot);
  this->s_suf = s_suf;

  vec3_t suf = (s_suf == FOOT_R ? rfoot : lfoot);
  vec3_t swf = (s_suf == FOOT_R ? lfoot : rfoot);
  float sign = (s_suf == FOOT_R ? 1.0f : -1.0f);

  // round to support foot coord.
  st.swf.x() =      ( swf.x() - suf.x() );
  st.swf.y() = sign*( swf.y() - suf.y() );
  st.swf.z() =      ( swf.z() );
  st.icp.x() =      ( icp.x() - suf.x() );
  st.icp.y() = sign*( icp.y() - suf.y() );

  st       = grid->roundState(st).state;
  state_id = grid->roundState(st).id;
}

void Search::setReference(arr2_t posRef, arr2_t icpRef){
  this->posRef.resize(posRef.size() );
  this->icpRef.resize(icpRef.size() );

  for(size_t i = 0; i < posRef.size(); i++) {
    this->posRef[i].x() =      ( posRef[i].x() - suf.x() );
    this->posRef[i].y() = sign*( posRef[i].y() - suf.y() );
    this->icpRef[i].x() =      ( icpRef[i].x() - suf.x() );
    this->icpRef[i].y() = sign*( icpRef[i].y() - suf.y() );
  }

  // g_foot.x() = round(g_foot.x() / 0.05) * 0.05;
  // g_foot.y() = round(g_foot.y() / 0.05) * 0.05;
}

bool Search::calc(int preview){
  tree->clear();
  seq.clear();
  region.clear();

  g_node = tree->search(state_id, s_suf, posRef, icpRef, preview);
  if(g_node == NULL)
    return false;

  calcSequence();
  region = tree->getCaptureRegion(s_state_id, s_input_id, s_suf, suf);

  return true;
}

Trans Search::getTrans(){
  Trans trans;

  std::vector<int> input_ids;
  Node            *node = g_node;
  while(node != NULL) {
    trans.states.push_back(grid->getState(node->state_id) );
    trans.inputs.push_back(grid->getInput(node->input_id) );
    input_ids.push_back(node->input_id);

    node = node->parent;
  }
  trans.inputs.pop_back();
  input_ids.pop_back();

  std::reverse(trans.states.begin(), trans.states.end() );
  std::reverse(trans.inputs.begin(), trans.inputs.end() );
  std::reverse(input_ids.begin(), input_ids.end() );

  s_input_id = input_ids[0];

  return trans;
}

State Search::getState(){
  return ini_state;
}

void Search::calcSequence(){
  Trans trans = getTrans();

  vec2_t   suf_pos(0, 0);
  vec2_t   swf_pos(0, 0);
  vec2_t   icp_pos(0, 0);
  vec2_t   cop_pos(0, 0);
  Sequence seq_;

  int amari;
  if(s_suf == Foot::FOOT_R) {
    amari   = 0;
    suf_pos = rfoot;
  }else{
    amari   = 1;
    suf_pos = lfoot;
  }

  for(size_t i = 0; i < trans.states.size(); i++) { // right support
    if( ( (int)i % 2 ) == amari) {
      icp_pos = suf_pos + trans.states[i].icp;
      if(i < trans.inputs.size() ) {
        cop_pos = suf_pos + trans.inputs[i].cop;
      }
      seq_.substitute(Foot::FOOT_R, suf_pos, icp_pos, cop_pos);

      if(i == 0) {
        ini_state.icp = suf_pos + trans.states[0].icp;
        ini_state.swf = vec2Tovec3(suf_pos) + trans.states[0].swf;
        ini_input.cop = suf_pos + trans.inputs[0].cop;
        ini_input.swf = suf_pos + trans.inputs[0].swf;
      }

      suf_pos = suf_pos + trans.inputs[i].swf;
    }else{ // left support
      icp_pos = suf_pos + mirror(trans.states[i].icp);
      if(i < trans.inputs.size() ) {
        cop_pos = suf_pos + mirror(trans.inputs[i].cop );
      }
      seq_.substitute(Foot::FOOT_L, suf_pos, icp_pos, cop_pos);

      if(i == 0) {
        ini_state.icp = suf_pos + mirror(trans.states[0].icp);
        ini_state.swf = vec2Tovec3(suf_pos) + mirror(trans.states[0].swf);
        ini_input.cop = suf_pos + mirror(trans.inputs[0].cop);
        ini_input.swf = suf_pos + mirror(trans.inputs[0].swf);
      }

      suf_pos = suf_pos + mirror(trans.inputs[i].swf);
    }
    seq.push_back(seq_);
  }
  // seq.erase(seq.begin() );
}

std::vector<Sequence> Search::getSequence(){
  return seq;
}

arr3_t Search::getFootstepR(){
  int offset = 0;
  if(s_suf == Foot::FOOT_R) {
    offset = 1;
  }

  arr3_t footstep_r;
  for(size_t i = offset; i < seq.size(); i++) {
    if(seq[i].suf == Foot::FOOT_R) {
      footstep_r.push_back(seq[i].pos);
    }
  }
  return footstep_r;
}

arr3_t Search::getFootstepL(){
  int offset = 0;
  if(s_suf == Foot::FOOT_L) {
    offset = 1;
  }

  arr3_t footstep_l;
  for(size_t i = offset; i < seq.size(); i++) {
    if(seq[i].suf == Foot::FOOT_L) {
      footstep_l.push_back(seq[i].pos);
    }
  }
  return footstep_l;
}

std::vector<CaptData> Search::getCaptureRegion(){
  return region;
}

#endif
} // namespace Capt