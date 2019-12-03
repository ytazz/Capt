#include "planner.h"

using namespace Capt;

Planner::Planner(Model *model, Param *param, Grid *grid, Capturability *capt){
  tree   = new Tree(param, grid, capt);
  search = new Search(grid, tree);
}

Planner::~Planner(){
  delete tree;
  delete search;
}

void Planner::set(planner::Input input){
  this->input = input;

  if(input.elapsed_time < dt) {
    selectSupportFoot();
    runSearch();
    generatePath();
  }
}

planner::Output Planner::get(){
  return this->output;
}

arr3_t Planner::getFootstepR(){
  return search->getFootstepR();
}

arr3_t Planner::getFootstepL(){
  return search->getFootstepL();
}

void Planner::selectSupportFoot(){
  switch (input.suf) {
  case FOOT_NONE:
    this->suf = FOOT_R;
    break;
  case FOOT_R:
    this->suf = FOOT_R;
    break;
  case FOOT_L:
    this->suf = FOOT_L;
    break;
  }
}

void Planner::runSearch(){
  vec2_t rfoot(input.rfoot.x(), input.rfoot.y() );
  vec2_t lfoot(input.lfoot.x(), input.lfoot.y() );
  vec2_t icp(input.icp.x(), input.icp.y() );
  vec2_t goal(input.goal.x(), input.goal.y() );
  search->setStart(rfoot, lfoot, icp, suf);
  search->setGoal(goal, input.stance);
  search->calc();
}

void Planner::generatePath(){

}