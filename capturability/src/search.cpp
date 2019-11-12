#include "search.h"

namespace Capt {

Search::Search(Grid *grid, Capturability *capturability) :
  grid(grid), capturability(capturability){
  gridmap = new GridMap(param);

  max_step = capturability->getMaxStep();
}

Search::~Search() {
}

void Search::clear(){
  opens.clear();
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

void Search::addOpen(Node* node){
  opens.push_back(node);
}

void Search::openNode(int node_id){
  Node *node = opens[node_id];

  vec2_t pos;
  for(int n = 1; n <= max_step; n++) {
    std::vector<CaptureSet> region = capturability->getCaptureRegion(node->state_id, n);
    for(size_t i = 0; i < region.size(); i++) {
      pos.x() = node->pos.x() + grid->getInput(region.input_id).x();
      if( ( node->step % 2 ) == 0) { // 最初の支持足と同じ足
        if(s_suf == FOOT_R)
          pos.y() = node->pos.y() + grid->getInput(region.input_id).y();
        else
          pos.y() = node->pos.y() - grid->getInput(region.input_id).y();
      }else{ // 最初の支持足と逆の足
        if(s_suf == FOOT_R)
          pos.y() = node->pos.y() - grid->getInput(region.input_id).y();
        else
          pos.y() = node->pos.y() + grid->getInput(region.input_id).y();
      }
      if(gridmap->getOccupancy() == OccupancyType::NONE) {
        Node node_;
        node_.parent   = node;
        node_.state_id = region[i].next_id;
        node_.step     = node->step + 1;

        gridmap->setNode(pos, node_);
        addOpen(gridmap->getNode(pos) );
      }
    }
  }

  opens.erase(opens.begin() + node_id);
}

bool Search::existOpen(){
  bool flag = false;
  if(opens.size() > 0)
    flag = true;
  return flag;
}

void Search::exe(){
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
  nodes.push_back(node);

  // Add open list
  addOpen(0);

  // Search
  // while(existOpen() ) {
  //   openNode(opens[0]);
  // }
}

void Search::step(){
  printf("-----------------------------\n");
  printf("opens: %d\n", (int)opens.size() );
  printf("nodes: %d\n", (int)nodes.size() );
  printf("\t| state_id | step |\n");
  for(size_t i = 0; i < nodes.size(); i++)
    printf("\t| %8d | %4d |\n", nodes[i].state_id, nodes[i].step);

  if(existOpen() ) {
    openNode(opens[0]);
  }
}

} // namespace Capt