#include "search.h"

namespace Capt {

Search::Search(GridMap *gridmap, Grid *grid, Capturability *capturability) :
  gridmap(gridmap), grid(grid), capturability(capturability){
  max_step = capturability->getMaxStep();
}

Search::~Search() {
}

void Search::clear(){
}

void Search::setStart(vec2_t rfoot, vec2_t lfoot, vec2_t icp, Foot suf){
  s_rfoot = rfoot;
  s_lfoot = lfoot;
  s_icp   = icp;
  s_suf   = suf;
}

void Search::setGoal(vec2_t rfoot, vec2_t lfoot){
  g_rfoot = rfoot;
  g_lfoot = lfoot;
}

void Search::open(Cell *cell){
  vec2_t pos;
  for(int n = 1; n <= max_step; n++) {
    std::vector<CaptureSet> region = capturability->getCaptureRegion(cell->node.state_id, n);
    for(size_t i = 0; i < region.size(); i++) {
      pos.x() = cell->pos.x() + grid->getInput(region[i].input_id).swf.x();
      if( ( cell->node.step % 2 ) == 0) { // 最初の支持足と同じ足
        if(s_suf == FOOT_R)
          pos.y() = cell->pos.y() + grid->getInput(region[i].input_id).swf.y();
        else
          pos.y() = cell->pos.y() - grid->getInput(region[i].input_id).swf.y();
      }else{ // 最初の支持足と逆の足
        if(s_suf == FOOT_R)
          pos.y() = cell->pos.y() - grid->getInput(region[i].input_id).swf.y();
        else
          pos.y() = cell->pos.y() + grid->getInput(region[i].input_id).swf.y();
      }
      if(gridmap->getOccupancy(pos) == OccupancyType::EMPTY) {
        vec2_t diff = pos - cell->pos;

        Node node_;
        node_.parent   = &cell->node;
        node_.state_id = region[i].next_id;
        node_.cost     = sqrt(diff.x() * diff.x() + diff.y() * diff.y() );
        node_.step     = cell->node.step + 1;

        gridmap->setNode(pos, node_);
      }
    }
  }
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
    state.swf = s_lfoot - s_rfoot;
    state.icp = s_icp - s_rfoot;
  }else{
    state.swf = s_rfoot - s_lfoot;
    state.icp = s_icp - s_lfoot;
  }

  // Calculate initial node
  Node node;
  node.parent   = NULL;
  node.state_id = grid->getStateIndex(state);
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
    open(cell);
    cell->type = OccupancyType::CLOSED;
    // gridmap->plot();
    return true;
  }else{
    return false;
  }
}

} // namespace Capt