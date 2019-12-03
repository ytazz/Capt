#include "planner.h"

using namespace Capt;

Planner::Planner(Model *model, Param *param, Grid *grid, Capturability *capt){
  tree   = new Tree(param, grid, capt);
  search = new Search(grid, tree);
}

Planner::Planner(){
  delete tree;
  delete search;
}

void Planner::set(planner::Input input){
  this->input = input;
}

planner::Output Planner::get(){
  return this->output;
}