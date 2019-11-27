#include "search.h"

namespace Capt {

Search::Search(Grid *grid, Capturability *capturability) :
  grid(grid), capturability(capturability){
  max_step = capturability->getMaxStep();
  g_node   = NULL;
  clear();
}

Search::~Search() {
}

void Search::clear(){
  num_node = 0;
  for(int i = 0; i < MAX_NODE_SIZE; i++) {
    nodes[i].parent   = NULL;
    nodes[i].state_id = 0;
    nodes[i].input_id = 0;
    nodes[i].pos << 0, 0;
    nodes[i].cost = 0.0;
    nodes[i].step = 0;
  }
}

void Search::setStanceWidth(double stance){
  this->stance = stance;
}

void Search::setStart(vec2_t rfoot, vec2_t lfoot, vec2_t icp, Foot suf){
  this->rfoot = rfoot;
  this->lfoot = lfoot;
  s_suf       = suf;

  if(s_suf == FOOT_R) {
    s_rfoot << 0, 0;
    s_lfoot = lfoot - rfoot;
    s_icp   = icp - rfoot;
  }
}

void Search::setGoal(vec2_t center){
  // world coordinate
  g_rfoot.x() = center.x();
  g_rfoot.y() = center.y() - stance / 2.0;
  g_lfoot.x() = center.x();
  g_lfoot.y() = center.y() + stance / 2.0;

  // initial support foot coordinate
  if(s_suf == FOOT_R) {
    g_rfoot -= rfoot;
    g_lfoot -= rfoot;
  }
}

Node* Search::findMinCostNode(){
  // printf("open list %3d\n", (int)opens.size() );
  double min  = 100;
  int    id   = 0;
  Node  *node = NULL;
  for(size_t i = 0; i < opens.size(); i++) {
    if(opens[i]->cost < min) {
      node = opens[i];
      id   = (int)i;
      min  = opens[i]->cost;
    }
  }
  // printf("min cost %lf\n", min);
  opens.erase(opens.begin() + id);
  return node;
}

bool Search::open(Node *node){
  int    num_step = node->step + 1;
  vec2_t swf;
  vec2_t next_pos;
  double cost = 0.0;

  if(num_step > 7)
    return false;

  for(int n = 1; n <= max_step; n++) {
    std::vector<CaptureSet> region = capturability->getCaptureRegion(node->state_id, n);
    for(size_t i = 0; i < region.size(); i++) {
      swf = grid->getInput(region[i].input_id).swf;
      if( ( num_step % 2 ) == 1) { // 最初の支持足と同じ方
        next_pos.x() = node->pos.x() + swf.x();
        next_pos.y() = node->pos.y() + swf.y();
        cost         = ( g_lfoot - next_pos ).norm();
      }else{ // 最初の支持足と逆の方
        next_pos.x() = node->pos.x() + swf.x();
        next_pos.y() = node->pos.y() - swf.y();
        cost         = ( g_rfoot - next_pos ).norm();
      }
      nodes[num_node].parent   = node;
      nodes[num_node].state_id = region[i].next_id;
      nodes[num_node].input_id = region[i].input_id;
      nodes[num_node].pos      = next_pos;
      nodes[num_node].cost     = cost;
      nodes[num_node].step     = num_step;
      if(cost < 0.01) {
        printf("find !\n");
        g_node = &nodes[num_node];
        return true;
      }
      opens.push_back(&nodes[num_node]);
      num_node++;
    }
  }

  // Node::printItemWithPos();
  // for(int i = 0; i < num_node; i++) {
  //   nodes[i].printWithPos();
  // }

  return false;
}

void Search::init(){
  // Calculate initial state
  State state;
  if(s_suf == FOOT_R) {
    state.swf = s_lfoot;
    state.icp = s_icp;
  }
  state.print();

  // Calculate cost
  double cost =  ( s_rfoot - g_rfoot ).norm();

  // Calculate initial node
  nodes[num_node].parent   = NULL;
  nodes[num_node].state_id = grid->getStateIndex(state);
  nodes[num_node].cost     = cost;
  nodes[num_node].step     = 0;
  num_node++;

  opens.push_back(&nodes[0]);
  Node::printItemWithPos();
  nodes[0].printWithPos();
}

void Search::exe(){
  // Set initial node
  init();

  // Search
  int i = 1;
  while(step() ) {
    // printf("%d-th\n", i);
    i++;
  }

  calcFootstep();
}

bool Search::step(){
  if(opens.size() > 0) {
    Node *node = findMinCostNode();
    if(open(node) ) {
      printf("found solution !\n");
      return false;
    }
    return true;
  }else{
    printf("cannot found solution ...\n");
    return false;
  }
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