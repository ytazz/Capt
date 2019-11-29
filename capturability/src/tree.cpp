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

void Tree::generate(){
  Node   node_;
  vec2_t pos;

  // set start nodes
  // for(int state_id = 0; state_id < state_num; state_id++) {
  for(int state_id = 24371; state_id < 24372; state_id++) {
    node_.parent   = NULL;
    node_.state_id = state_id;
    node_.input_id = 0;
    node_.step     = 0;
    node_.pos << 0.0, 0.0;

    auto region = capturability->getCaptureRegion(state_id);
    if(region.size() > 0) {
      nodes[num_node] = node_;
      num_node++;
    }
  }
  for(int i = 0; i < num_node; i++) {
    opens.push_back(&nodes[i]);
  }

  // calculate reaves
  int count = 0;
  while(opens.size() > 0) {
    // while(count < 2) {
    // printf("nodes: %8d\t", num_node );
    // printf("opens: %8d\n", (int)opens.size() );
    Node* target = opens[0];
    if(target->step < stepMax) {
      auto region = capturability->getCaptureRegion(target->state_id);
      for(size_t i = 0; i < region.size(); i++) {
        // calculate next landing position
        vec2_t base = target->pos;
        vec2_t swf  = grid->getInput(region[i].input_id).swf;
        if(target->step % 2 == 0) { // if right foot support
          pos.x() = base.x() + swf.x();
          pos.y() = base.y() + swf.y();
        }else{ // if left foot support
          pos.x() = target->pos.x() + swf.x();
          pos.y() = target->pos.y() - swf.y();
        }

        // set
        node_.parent   = target;
        node_.state_id = region[i].next_id;
        node_.input_id = region[i].input_id;
        node_.step     = target->step + 1;
        node_.pos      = pos;
        if(gridMap->getOccupancy(pos) == OccupancyType::EMPTY) {
          nodes[num_node] = node_;
          opens.push_back(&nodes[num_node] );
          if(region[i].nstep == 1)
            gridMap->setNode(pos, &nodes[num_node] );
          num_node++;
        }
      }
    }
    opens.erase(opens.begin() );
    count++;
  }

  // gridMap->plot();
}

Node* Tree::getReafNode(int state_id, vec2_t pos){
  return gridMap->getNode(pos);
}

int Tree::getPreviewStep(){
  return stepMax;
}

} // namespace Capt