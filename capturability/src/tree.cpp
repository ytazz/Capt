#include "tree.h"

namespace Capt {

Tree::Tree(Grid *grid, Capturability* capturability) :
  grid(grid), capturability(capturability), epsilon(0.01){
  clear();
}

Tree::~Tree(){
}

void Tree::clear(){
  num_node = 0;
  opened   = 0;
  for(int i = 0; i < MAX_NODE_SIZE; i++) {
    nodes[i].parent   = NULL;
    nodes[i].state_id = 0;
    nodes[i].input_id = 0;
    nodes[i].suf      = FOOT_NONE;
    nodes[i].pos      = vec2_t(0, 0);
    nodes[i].step     = 0;
    nodes[i].err      = 0;
  }
}

Node* Tree::search(int state_id, Foot s_suf, arr2_t g_foot, int preview){
  // set start node
  nodes[num_node].state_id = state_id;
  nodes[num_node].suf      = s_suf;
  num_node++;

  // set n in each step
  int nStepCaptureBasin = -1;
  for(int i = 4; i > 0; i--) {
    if(capturability->capturable(state_id, i) ) {
      nStepCaptureBasin = i;
    }
  }
  if(nStepCaptureBasin < 0) {
    printf("NOT capturable\n");
    return NULL;
  }
  printf("%d-step capturable \n", nStepCaptureBasin);
  int n[10];
  for(int i = 0; i < 10; i++) {
    int n_ = nStepCaptureBasin - i + 1;
    if( n_ > 0) {
      n[i] = n_;
    }else{
      n[i] = 1;
    }
  }

  // calculate reaves
  vec2_t swf, pos;
  Node  *goal = NULL;
  double min  = 100;
  while(true) {
    // set target node based on breadth first search
    Node *target  = &nodes[opened];
    int   numStep = target->step + 1; // 何歩目か

    // target node expansion
    std::vector<CaptureSet*> region = capturability->getCaptureRegion(target->state_id, n[numStep]);
    for(size_t i = 0; i < region.size(); i++) {
      // set parent
      nodes[num_node].parent   = target;
      nodes[num_node].state_id = region[i]->next_id;
      nodes[num_node].input_id = region[i]->input_id;
      nodes[num_node].step     = numStep;
      // printf("-----------------------------------------\n");
      // printf(" %d\n", nodes[num_node].step);

      if(nodes[num_node].step > preview) {
        return goal;
      }

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

      nodes[num_node].err = target->err + ( nodes[num_node].pos - g_foot[nodes[num_node].step] ).norm();
      // determine if next position reach goal
      // if(nodes[num_node].suf == g_suf) {
      //   if( ( nodes[num_node].pos - g_foot ).norm() < epsilon) {
      //     return &nodes[num_node];
      //   }
      // }

      if(nodes[num_node].step == preview) {
        if(nodes[num_node].err < min) {
          goal = &nodes[num_node];
          min  = nodes[num_node].err;
        }
      }

      num_node++;
      if(num_node > MAX_NODE_SIZE - 100) {
        printf("reach max node size\n");
        return NULL;
      }
    }

    opened++;
  }
  return NULL;
}

std::vector<CaptData> Tree::getCaptureRegion(int state_id, int input_id, Foot suf, vec2_t p_suf){
  std::vector<CaptData>    region;
  std::vector<CaptureSet*> set;

  // get capture region
  set = capturability->getCaptureRegion(state_id);

  // substitute region variable
  region.resize(set.size() );
  for(size_t i = 0; i < region.size(); i++) {
    vec2_t swf = grid->getInput(set[i]->input_id).swf;
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
    region[i].nstep = set[i]->nstep;
  }

  return region;
}

} // namespace Capt