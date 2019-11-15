#ifndef __NODE_H__
#define __NODE_H__

#include "base.h"
#include <iostream>
#include <string>

namespace Capt {

struct Node {
  Node(){
  }

  static void printItem(){
    printf("| -------- | ------ | ------ | ------ | ---- |\n");
    printf("| state_id | g_cost | h_cost |  cost  | step |\n");
    printf("| -------- | ------ | ------ | ------ | ---- |\n");
  }

  static void printItemWithPos(){
    printf("| ----- | ----- | -------- | ------ | ------ | ------ | ---- |\n");
    printf("| pos_x | pos_y | state_id | g_cost | h_cost |  cost  | step |\n");
    printf("| ----- | ----- | -------- | ------ | ------ | ------ | ---- |\n");
  }

  void print(){
    printf("| %8d ", state_id);
    printf("| %2.4lf ", g_cost);
    printf("| %2.4lf ", h_cost);
    printf("| %2.4lf ", cost);
    printf("| %4d |\n", step);
  }

  void print(vec2_t pos){
    printf("| %+1.2lf ", pos.x() );
    printf("| %+1.2lf ", pos.y() );
    printf("| %8d ", state_id);
    printf("| %2.4lf ", g_cost);
    printf("| %2.4lf ", h_cost);
    printf("| %2.4lf ", cost);
    printf("| %4d |\n", step);
  }

  Node * parent;
  int    state_id;
  double g_cost;
  double h_cost;
  double cost;
  int    step;

  void operator=(const Node &node) {
    this->parent   = node.parent;
    this->state_id = node.state_id;
    this->g_cost   = node.g_cost;
    this->h_cost   = node.h_cost;
    this->cost     = node.cost;
    this->step     = node.step;
  }
};

} // namespace Capt

#endif // __NODE_H__