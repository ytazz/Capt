#include "tree.h"

namespace Capt {

Tree::Tree(Grid *grid, Capturability* capturability) :
  grid(grid), capturability(capturability), epsilon(0.05){
  clear();
}

Tree::~Tree(){
}

void Tree::clear(){
  num_node = 0;
  for(int i = 0; i < MAX_NODE_SIZE; i++) {
    nodes[i].parent   = NULL;
    nodes[i].state_id = 0;
    nodes[i].input_id = 0;
    nodes[i].suf      = FOOT_NONE;
    nodes[i].pos << 0, 0;
  }
  opened = 0;
}

Node* Tree::search(int state_id, Foot s_suf, vec2_t g_foot, Foot g_suf){
  // set start node
  nodes[num_node].parent   = NULL;
  nodes[num_node].state_id = state_id;
  nodes[num_node].input_id = 0;
  nodes[num_node].suf      = s_suf;
  nodes[num_node].pos << 0.0, 0.0;
  num_node++;

  // calculate reaves
  vec2_t swf, pos;
  while(true) {
    // set target node based on breadth first search
    Node *target = &nodes[opened];
    // target node expansion
    std::vector<CaptureSet*> region = capturability->getCaptureRegion(target->state_id, 1);
    for(size_t i = 0; i < region.size(); i++) {
      // set parent
      nodes[num_node].parent   = target;
      nodes[num_node].state_id = region[i]->next_id;
      nodes[num_node].input_id = region[i]->input_id;

      // calculate next landing position
      swf = grid->getInput(region[i]->input_id).swf;
      if(target->suf == FOOT_R) {   // if right foot support
        nodes[num_node].pos.x() = target->pos.x() + swf.x();
        nodes[num_node].pos.y() = target->pos.y() + swf.y();
        nodes[num_node].suf     = FOOT_L;
      }else{   // if left foot support
        nodes[num_node].pos.x() = target->pos.x() + swf.x();
        nodes[num_node].pos.y() = target->pos.y() - swf.y();
        nodes[num_node].suf     = FOOT_R;
      }

      // determine if next position reach goal
      if(nodes[num_node].suf == g_suf) {
        if( ( nodes[num_node].pos - g_foot ).norm() < epsilon) {
          return &nodes[num_node];
        }
      }

      num_node++;
    }
    opened++;
  }
  return NULL;
}

std::vector<CaptData> Tree::getCaptureRegion(int state_id, int input_id, Foot suf, vec2_t p_suf){
  std::vector<CaptData>    region;
  std::vector<CaptureSet*> set, set_;

  // calculate cop id
  int cop_id = grid->indexCop(grid->getInput(input_id).cop);

  // get capture region (all cop)
  set = capturability->getCaptureRegion(state_id);

  // extract designated capture region
  for(size_t i = 0; i < set.size(); i++) {
    if(grid->indexCop(grid->getInput(set[i]->input_id).cop) == cop_id)
      set_.push_back(set[i]);
  }

  // substitute region variable
  region.resize(set_.size() );
  for(size_t i = 0; i < region.size(); i++) {
    vec2_t swf = grid->getInput(set_[i]->input_id).swf;
    vec3_t pos;
    if(suf == FOOT_R) {
      pos.x() = p_suf.x() + swf.x();
      pos.y() = p_suf.y() + swf.y();
      pos.z() = 0.0;
    }else{
      pos.x() = p_suf.x() + swf.x();
      pos.y() = p_suf.y() - swf.y();
      pos.z() = 0.0;
    }
    region[i].pos   = pos;
    region[i].nstep = set_[i]->nstep;
  }

  return region;
}

} // namespace Capt