#include "tree.h"

namespace Capt {

Tree::Tree(Param *param, Grid *grid, Capturability* capturability) :
  grid(grid), capturability(capturability){
  gridMap = new GridMap(param);
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
    nodes[i].pos << 0, 0;
    nodes[i].step = 0;
  }
}

Node* Tree::search(int state_id, Foot suf, vec2_t g_rfoot, vec2_t g_lfoot){
  int amari;
  if(suf == FOOT_R) {
    amari = 0;
  }else{
    amari = 1;
  }

  // set start node
  nodes[num_node].parent   = NULL;
  nodes[num_node].state_id = state_id;
  nodes[num_node].input_id = 0;
  nodes[num_node].step     = 0;
  nodes[num_node].cost     = 0;
  nodes[num_node].pos << 0.0, 0.0;
  num_node++;

  opens.push_back(&nodes[0]);

  // calculate reaves
  vec2_t pos;
  while(opens.size() > 0) {
    // find minimun cost node
    double min = 100;
    int    id  = 0;
    for(size_t i = 0; i < opens.size(); i++) {
      if(opens[i]->cost < min) {
        id  = (int)i;
        min = opens[i]->cost;
      }
    }
    Node *target = opens[id];
    // target node expansion
    std::vector<CaptureSet*> region = capturability->getCaptureRegion(target->state_id);
    for(size_t i = 0; i < region.size(); i++) {
      // calculate next landing position
      vec2_t swf  = grid->getInput(region[i]->input_id).swf;
      double cost = 0.0;
      if(target->step % 2 == amari) {   // if right foot support
        pos.x() = target->pos.x() + swf.x();
        pos.y() = target->pos.y() + swf.y();
        cost    = ( g_lfoot - pos ).norm();
      }else{   // if left foot support
        pos.x() = target->pos.x() + swf.x();
        pos.y() = target->pos.y() - swf.y();
        cost    = ( g_rfoot - pos ).norm();
      }

      // determine if next position reach goal or not
      if(cost < 0.01) {
        nodes[num_node].parent   = target;
        nodes[num_node].state_id = region[i]->next_id;
        nodes[num_node].input_id = region[i]->input_id;
        nodes[num_node].step     = target->step + 1;
        nodes[num_node].cost     = cost;
        nodes[num_node].pos      = pos;

        // gridMap->plot();
        return &nodes[num_node];
      }else if(gridMap->getOccupancy(pos) != OccupancyType::NONE) {
        // set
        nodes[num_node].parent   = target;
        nodes[num_node].state_id = region[i]->next_id;
        nodes[num_node].input_id = region[i]->input_id;
        nodes[num_node].step     = target->step + 1;
        nodes[num_node].cost     = cost + 0.1 * nodes[num_node].step;
        nodes[num_node].pos      = pos;

        gridMap->setNode(pos, &nodes[num_node]);
        opens.push_back(&nodes[num_node]);
        num_node++;
      }
    }
    opens.erase(opens.begin() + id);
    // gridMap->plot();
    // usleep(0.5 * 1000000);
  }
  return NULL;
}

} // namespace Capt