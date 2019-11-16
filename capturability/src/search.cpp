#include "search.h"

namespace Capt {

Search::Search(GridMap *gridmap, Grid *grid, Capturability *capturability) :
  gridmap(gridmap), grid(grid), capturability(capturability){
  max_step = capturability->getMaxStep();
  g_node   = NULL;
  h_scale  = 0.1;
}

Search::~Search() {
}

void Search::clear(){
}

void Search::setStanceWidth(double stance){
  this->stance = stance;
}

void Search::setStart(vec2_t rfoot, vec2_t lfoot, vec2_t icp, Foot suf){
  s_rfoot = rfoot;
  s_lfoot = lfoot;
  s_icp   = icp;
  s_suf   = suf;

  if(s_suf == FOOT_R) {
    s_lfoot -= s_rfoot;
    s_icp   -= s_rfoot;
  }else{
    s_rfoot -= s_lfoot;
    s_icp   -= s_lfoot;
    s_rfoot *= -1;
    s_icp   *= -1;
  }
}

void Search::setGoal(vec2_t center){
  g_rfoot.x() = center.x();
  g_rfoot.y() = center.y() - stance / 2.0;
  g_lfoot.x() = center.x();
  g_lfoot.y() = center.y() + stance / 2.0;

  if(s_suf == FOOT_R) {
    g_rfoot -= s_rfoot;
    g_lfoot -= s_rfoot;
  }else{
    g_rfoot -= s_lfoot;
    g_lfoot -= s_lfoot;
  }

  gridmap->setOccupancy(g_rfoot, OccupancyType::GOAL);
  gridmap->setOccupancy(g_lfoot, OccupancyType::GOAL);
}

bool Search::open(Cell *cell){
  vec2_t pos;
  int    num_step = cell->node.step + 1;
  vec2_t swf;
  int    rl = ( num_step % 2 ); // right or left

  for(int n = 1; n <= max_step; n++) {
    std::vector<CaptureSet> region = capturability->getCaptureRegion(cell->node.state_id, n);
    for(size_t i = 0; i < region.size(); i++) {
      swf     = grid->getInput(region[i].input_id).swf;
      pos.x() = cell->pos.x() + swf.x();
      pos.y() = cell->pos.y() + yaxis[rl] * swf.y();
      if(gridmap->getOccupancy(pos) == OccupancyType::GOAL) {
        Node node_;
        node_.parent   = &cell->node;
        node_.state_id = region[i].next_id;
        node_.input_id = region[i].input_id;
        node_.step     = num_step;
        gridmap->setNode(pos, node_);

        g_node = gridmap->getNode(pos);

        return true;
      }
      if(gridmap->getOccupancy(pos) == OccupancyType::EMPTY) {
        double g_cost = cell->node.g_cost + ( pos - cell->pos ).norm();
        double h_cost = h_scale * ( pos - g_arr[rl] ).norm();

        Node node_;
        node_.parent   = &cell->node;
        node_.state_id = region[i].next_id;
        node_.input_id = region[i].input_id;
        node_.g_cost   = g_cost;
        node_.h_cost   = h_cost;
        node_.cost     = g_cost + h_cost;
        node_.step     = num_step;

        gridmap->setNode(pos, node_);
      }
    }
  }

  return false;
}

bool Search::existOpen(){
  bool flag = false;
  // if(opens.size() > 0)
  //   flag = true;
  return flag;
}

void Search::init(){
  // Calculate initial state
  State state;

  if(s_suf == FOOT_R) {
    state.swf = s_lfoot;
    state.icp = s_icp;

    s_arr[0] = s_rfoot;
    s_arr[1] = s_lfoot;
    g_arr[0] = g_rfoot;
    g_arr[1] = g_lfoot;
    yaxis[0] = -1;
    yaxis[1] = +1;
  }else{
    state.swf = s_rfoot;
    state.icp = s_icp;

    s_arr[0] = s_lfoot;
    s_arr[1] = s_rfoot;
    g_arr[0] = g_lfoot;
    g_arr[1] = g_rfoot;
    yaxis[0] = +1;
    yaxis[1] = -1;
  }
  state.print();

  // Calculate cost
  double g_cost = 0.0;
  double h_cost = h_scale * ( s_arr[0] - g_arr[0] ).norm();

  // Calculate initial node
  Node node;
  node.parent   = NULL;
  node.state_id = grid->getStateIndex(state);
  node.g_cost   = g_cost;
  node.h_cost   = h_cost;
  node.cost     = g_cost + h_cost;
  node.step     = 0;

  gridmap->setNode(vec2_t(0, 0), node);

  // gridmap->plot();
}

void Search::exe(){
  // Set initial node
  init();

  // Search
  while(step() ) {
  }
}

bool Search::step(){
  Cell *cell = gridmap->findMinCostCell();
  if(cell != NULL) {
    if(open(cell) ) {
      printf("found solution!\n");
      return false;
    }
    cell->type = OccupancyType::CLOSED;
    gridmap->plot();
    return true;
  }else{
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

} // namespace Capt