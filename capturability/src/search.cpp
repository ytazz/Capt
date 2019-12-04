#include "search.h"

namespace Capt {

Search::Search(Grid *grid, Tree *tree) :
  grid(grid), tree(tree){
}

Search::~Search() {
}

void Search::clear(){
}

void Search::setStart(vec2_t rfoot, vec2_t lfoot, vec2_t icp, Foot suf){
  this->rfoot = rfoot;
  this->lfoot = lfoot;
  this->s_suf = suf;

  // round to support foot coord.
  if(s_suf == FOOT_R) {
    s_lfoot.x() = +( lfoot.x() - rfoot.x() );
    s_lfoot.y() = +( lfoot.y() - rfoot.y() );
    s_icp.x()   = +( icp.x() - rfoot.x() );
    s_icp.y()   = +( icp.y() - rfoot.y() );
    s_state.icp = s_icp;
    s_state.swf = s_lfoot;
  }else{
    s_rfoot.x() = +( rfoot.x() - lfoot.x() );
    s_rfoot.y() = -( rfoot.y() - lfoot.y() );
    s_icp.x()   = +( icp.x() - lfoot.x() );
    s_icp.y()   = -( icp.y() - lfoot.y() );
    s_state.icp = s_icp;
    s_state.swf = s_rfoot;
  }
  s_state    = grid->roundState(s_state).state;
  s_state_id = grid->roundState(s_state).id;
}

void Search::setGoal(vec2_t center, double stance){
  // support foot coord.
  vec2_t suf_to_center;
  if(s_suf == FOOT_R) {
    suf_to_center = grid->roundVec(center - rfoot);
  }else{
    suf_to_center = grid->roundVec(center - lfoot);
  }
  g_rfoot.x() = suf_to_center.x();
  g_rfoot.y() = suf_to_center.y() - stance / 2;
  g_lfoot.x() = suf_to_center.x();
  g_lfoot.y() = suf_to_center.y() + stance / 2;
}

void Search::calc(){
  g_node = tree->search(s_state_id, s_suf, g_rfoot, g_lfoot);
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

State Search::getState(){
  return ini_state;
}

Input Search::getInput(){
  return ini_input;
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
    suf_pos = rfoot;
  }else{
    amari   = 1;
    suf_pos = lfoot;
  }

  footstep.clear();
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

  ini_state = trans.states[0];
  ini_input = trans.inputs[0];
}

std::vector<Footstep> Search::getFootstep(){
  return footstep;
}

arr3_t Search::getFootstepR(){
  arr3_t footstep_r;
  for(size_t i = 0; i < footstep.size(); i++) {
    if(footstep[i].suf == Foot::FOOT_R) {
      footstep_r.push_back(footstep[i].pos);
    }
  }
  return footstep_r;
}

arr3_t Search::getFootstepL(){
  arr3_t footstep_l;
  for(size_t i = 0; i < footstep.size(); i++) {
    if(footstep[i].suf == Foot::FOOT_L) {
      footstep_l.push_back(footstep[i].pos);
    }
  }
  return footstep_l;
}

} // namespace Capt