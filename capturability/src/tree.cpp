#include "tree.h"

namespace Capt {

Tree::Tree(Capturability* capturability, Grid *grid, Param *param) :
  capturability(capturability), grid(grid){
  captMax   = capturability->getMaxStep();
  stepMax   = 1;
  state_num = grid->getNumState();

  num_node = 0;
  for(int i = 0; i < MAX_NODE_SIZE; i++) {
    nodes[i].parent   = NULL;
    nodes[i].state_id = 0;
    nodes[i].input_id = 0;
    nodes[i].pos << 0, 0;
    nodes[i].step = 0;
  }

  gridMap = new GridMap(param);
}

Tree::~Tree(){
}

void Tree::setPreviewStep(int stepMax){
  this->stepMax = stepMax;
}

Node* Tree::search(int state_id, vec2_t g_foot){
  vec2_t g_rfoot, g_lfoot;
  g_rfoot.x() = g_foot.x();
  g_rfoot.y() = g_foot.y() - 0.2;
  g_lfoot.x() = g_foot.x();
  g_lfoot.y() = g_foot.y() + 0.2;

  vec2_t pos;

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
  while(opens.size() > 0) {
    // printf("nodes: %8d\t", num_node );
    // printf("opens: %8d\n", (int)opens.size() );
    double min = 100;
    int    id  = 0;
    for(size_t i = 0; i < opens.size(); i++) {
      if(opens[i]->cost < min) {
        id  = (int)i;
        min = opens[i]->cost;
      }
    }
    // printf("min cost id:%d val:%lf\n", id, min);
    Node                    *target = opens[id];
    std::vector<CaptureSet*> region = capturability->getCaptureRegion(target->state_id);
    for(size_t i = 0; i < region.size(); i++) {
      // calculate next landing position
      vec2_t swf  = grid->getInput(region[i]->input_id).swf;
      double cost = 0.0;
      if(target->step % 2 == 0) {   // if right foot support
        pos.x() = target->pos.x() + swf.x();
        pos.y() = target->pos.y() + swf.y();
        cost    = ( g_lfoot - pos ).norm();
      }else{   // if left foot support
        pos.x() = target->pos.x() + swf.x();
        pos.y() = target->pos.y() - swf.y();
        cost    = ( g_rfoot - pos ).norm();
      }

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
        nodes[num_node].cost     = cost;
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

int Tree::getPreviewStep(){
  return stepMax;
}

} // namespace Capt