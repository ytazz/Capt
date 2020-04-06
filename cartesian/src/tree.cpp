#include "tree.h"

namespace Capt {

Tree::Tree(Grid *grid, Capturability* capturability) :
  grid(grid), capturability(capturability), epsilon(0.01){
  for(int i = 0; i < MAX_NODE_SIZE; i++) {
    nodes[i].parent   = NULL;
    nodes[i].state_id = 0;
    nodes[i].input_id = 0;
    nodes[i].suf      = FOOT_NONE;
    nodes[i].pos      = vec2_t(0, 0);
    nodes[i].step     = 0;
    nodes[i].err      = 0;
  }
  clear();
}

Tree::~Tree(){
}

void Tree::clear(){
  num_node          = 0;
  opened            = 0;
  nodes[0].parent   = NULL;
  nodes[0].state_id = 0;
  nodes[0].input_id = 0;
  nodes[0].suf      = FOOT_NONE;
  nodes[0].pos      = vec2_t(0, 0);
  nodes[0].step     = 0;
  nodes[0].err      = 0;
}

Node* Tree::search(int state_id, Foot s_suf, arr2_t posRef, arr2_t icpRef, int preview){
  // set start node
  nodes[num_node].state_id = state_id;
  nodes[num_node].suf      = s_suf;
  num_node++;

  // set n in each step
  int nStepCaptureBasin = capturability->isCapturable(state_id, -1);
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

  int goalNode = 0;

  // calculate leaves
  vec2_t swf, pos, icp, icp_;
  Node  *goal   = NULL;
  float  min    = 100;
  float  posErr = 0.0, icpErr = 0.0;

  while(true) {
    // set target node based on breadth first search
    if(opened == num_node) {
      printf("no capture region...\n");
      return &( nodes[num_node - 1] );
    }

    Node *target  = &nodes[opened];
    int   numStep = target->step + 1; // 何歩目か

    // target node expansion
    if(numStep > preview)
      return goal;

    std::vector<CaptureSet*> region = capturability->getCaptureRegion(target->state_id, n[numStep]);
    // printf("parent state_id %6d\n", target->state_id);
    for(size_t i = 0; i < region.size(); i++) {

      // set parent
      // printf("     child  input_id %6d, ", region[i]->input_id);
      // printf("     child  state_id %6d\n", region[i]->next_id);
      nodes[num_node].parent   = target;
      nodes[num_node].state_id = region[i]->next_id;
      nodes[num_node].input_id = region[i]->input_id;
      nodes[num_node].step     = numStep;
      // printf("-----------------------------------------\n");
      // printf(" %d\n", nodes[num_node].step);

      // calculate next landing position
      swf = grid->getInput(region[i]->input_id).swf;
      icp = grid->getState(region[i]->next_id).icp;
      if(target->suf == FOOT_R) {   // if right foot support
        nodes[num_node].pos.x() = target->pos.x() + swf.x();
        nodes[num_node].pos.y() = target->pos.y() + swf.y();
        nodes[num_node].suf     = FOOT_L;
        icp_.x()                = nodes[num_node].pos.x() + icp.x();
        icp_.y()                = nodes[num_node].pos.x() - icp.y();
      }else{   // if left foot support
        nodes[num_node].pos.x() = target->pos.x() + swf.x();
        nodes[num_node].pos.y() = target->pos.y() - swf.y();
        nodes[num_node].suf     = FOOT_R;
        icp_.x()                = nodes[num_node].pos.x() + icp.x();
        icp_.y()                = nodes[num_node].pos.x() + icp.y();
      }

      // posErr = ( nodes[num_node].pos - posRef[nodes[num_node].step] ).norm();
      // icpErr = ( icp_ - icpRef[nodes[num_node].step] ).norm();
      posErr = 0;
      icpErr = icp.norm();

      nodes[num_node].err = target->err + posErr + icpErr;
      // determine if next position reach goal
      // if(nodes[num_node].suf == g_suf) {
      //   if( ( nodes[num_node].pos - g_foot ).norm() < epsilon) {
      //     return &nodes[num_node];
      //   }
      // }

      if(numStep == preview) {
        if(nodes[num_node].err < min) {
          goal     = &nodes[num_node];
          min      = nodes[num_node].err;
          goalNode = num_node;
        }
      }

      num_node++;
      if(num_node > MAX_NODE_SIZE - 100) {
        printf("reach max node size\n");
        return NULL;
      }
    }

    opened++;
    if(opened > MAX_NODE_SIZE - 1000) {
      printf("reach max opened size\n");
      return NULL;
    }
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
    region[i].pos   = p_suf + vec2_t(swf.x(), (suf == FOOT_R ? 1.0 : -1.0)*swf.y());;
    region[i].nstep = set[i]->nstep;
  }

  return region;
}

} // namespace Capt