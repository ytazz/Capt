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
    printf("| -------- | -------- | ------ | ---- |\n");
    printf("| state_id | input_id |  cost  | step |\n");
    printf("| -------- | -------- | ------ | ---- |\n");
  }

  static void printItemWithPos(){
    printf("| ----- | ----- | -------- | -------- | ------ | ---- |\n");
    printf("| pos_x | pos_y | state_id | input_id |  cost  | step |\n");
    printf("| ----- | ----- | -------- | -------- | ------ | ---- |\n");
  }

  void print(){
    printf("| %8d ", state_id);
    printf("| %8d ", input_id);
  }

  void printWithPos(){
    printf("| %+1.2lf ", pos.x() );
    printf("| %+1.2lf ", pos.y() );
    printf("| %8d ", state_id);
    printf("| %8d ", input_id);
  }

  Node * parent;
  int    state_id;
  int    input_id;
  Foot   suf;
  vec2_t pos;
  int    step;
  double err;

  void operator=(const Node &node) {
    this->parent   = node.parent;
    this->state_id = node.state_id;
    this->input_id = node.input_id;
    this->suf      = node.suf;
    this->pos      = node.pos;
    this->step     = node.step;
    this->err      = node.err;
  }
};

} // namespace Capt

#endif // __NODE_H__