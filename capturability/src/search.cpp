#include "search.h"

namespace Capt {

Search::Search(Grid *grid, Tree *tree) :
  grid(grid), tree(tree){
}

Search::~Search() {
}

void Search::setStart(vec2_t rfoot, vec2_t lfoot, vec2_t icp, Foot suf){
  this->rfoot = rfoot;
  this->lfoot = lfoot;
  this->s_suf = suf;

  if(s_suf == FOOT_R) {
    s_lfoot = lfoot - rfoot;
    s_icp   = icp - rfoot;
  }
}

void Search::setGoal(vec2_t center){
  if(s_suf == FOOT_R) {
    g_foot.x() = center.x() - rfoot.x();
    g_foot.y() = center.y() - rfoot.y();
  }else{
    g_foot.x() = center.x() - lfoot.x();
    g_foot.y() = -( center.y() - lfoot.y() );
  }
}

void Search::calc(){
  State state;
  if(s_suf == FOOT_R) {
    state.icp = s_icp;
    state.swf = s_lfoot;
  }
  g_node = tree->getReafNode(grid->getStateIndex(state), g_foot);
  calcFootstep();
}

Trans Search::getTrans(){
  Trans trans;
  trans.size = g_node->step;

  Node *node = g_node;
  // Node::printItem();
  while(node != NULL) {
    // node->print();

    trans.states.push_back(grid->getState(node->state_id) );
    trans.inputs.push_back(grid->getInput(node->input_id) );

    node = node->parent;
  }
  trans.inputs.pop_back();

  std::reverse(trans.states.begin(), trans.states.end() );
  std::reverse(trans.inputs.begin(), trans.inputs.end() );

  return trans;
}

void Search::calcFootstep(){
  Trans trans = getTrans();

  vec2_t   suf_pos(0, 0);
  vec2_t   swf_pos(0, 0);
  vec2_t   icp_pos(0, 0);
  vec2_t   cop_pos(0, 0);
  Footstep footstep_;

  int amari;
  if(s_suf == Foot::FOOT_R) {
    amari   = 0;
    suf_pos = s_rfoot;
  }else{
    amari   = 1;
    suf_pos = s_lfoot;
  }

  for(size_t i = 0; i < trans.states.size(); i++) { // right support
    if( ( (int)i % 2 ) == amari) {
      icp_pos = suf_pos + trans.states[i].icp;
      if(i < trans.inputs.size() ) {
        cop_pos = suf_pos + trans.inputs[i].cop;
      }
      footstep_.substitute(Foot::FOOT_R, suf_pos, icp_pos, cop_pos);

      suf_pos = suf_pos + trans.inputs[i].swf;
    }else{ // left support
      icp_pos = suf_pos + mirror(trans.states[i].icp);
      if(i < trans.inputs.size() ) {
        cop_pos = suf_pos + mirror(trans.inputs[i].cop );
      }
      footstep_.substitute(Foot::FOOT_L, suf_pos, icp_pos, cop_pos);

      suf_pos = suf_pos + mirror(trans.inputs[i].swf);
    }
    footstep.push_back(footstep_);
  }
}

std::vector<Footstep> Search::getFootstep(){
  return footstep;
}

std::vector<vec3_t> Search::getFootstepR(){
  std::vector<vec3_t> footstep_r;
  for(size_t i = 0; i < footstep.size(); i++) {
    if(footstep[i].suf == Foot::FOOT_R) {
      footstep_r.push_back(footstep[i].pos);
    }
  }
  return footstep_r;
}

std::vector<vec3_t> Search::getFootstepL(){
  std::vector<vec3_t> footstep_l;
  for(size_t i = 0; i < footstep.size(); i++) {
    if(footstep[i].suf == Foot::FOOT_L) {
      footstep_l.push_back(footstep[i].pos);
    }
  }
  return footstep_l;
}

} // namespace Capt