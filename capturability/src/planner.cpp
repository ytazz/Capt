#include "planner.h"

using namespace Capt;

Planner::Planner(Model *model, Param *param, Config *config, Grid *grid, Capturability *capt){
  tree      = new Tree(param, grid, capt);
  search    = new Search(grid, tree);
  swingFoot = new SwingFoot(model);
  pendulum  = new Pendulum(model);

  param->read(&omega, "omega");
  config->read(&dt, "timestep");
}

Planner::~Planner(){
  delete tree;
  delete search;
}

void Planner::set(planner::Input input){
  this->input = input;

  if(input.elapsed < dt) {
    selectSupportFoot();
    runSearch();
  }

  generatePath();
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
  vec2_t rfoot = vec3Tovec2(input.rfoot);
  vec2_t lfoot = vec3Tovec2(input.lfoot);
  vec2_t icp   = vec3Tovec2(input.com + input.com_vel / omega);
  vec2_t goal  = vec3Tovec2(input.goal);
  search->setStart(rfoot, lfoot, icp, suf);
  search->setGoal(goal, input.stance);
  search->calc();

  // get state & input
  State s = search->getState();
  Input i = search->getInput();

  // calc swing foot trajectory
  swingFoot->set(s.swf, i.swf);

  // calc com & icp trajectory
  pendulum->setCop(i.cop);
  pendulum->setIcp(s.icp);
  pendulum->setCom(input.com);
  pendulum->setComVel(input.com_vel);

  // set to output
  output.duration = swingFoot->getTime();
}

void Planner::generatePath(){
  if(suf == FOOT_R) {
    output.rfoot = input.rfoot;
    output.lfoot = input.rfoot + swingFoot->getTraj(input.elapsed + dt);
  }else{
    output.rfoot = input.lfoot + mirror(swingFoot->getTraj(input.elapsed + dt) );
    output.lfoot = input.lfoot;
  }
  output.icp = vec2Tovec3(pendulum->getIcp(input.elapsed + dt) );
  output.cop = vec2Tovec3(pendulum->getCop(input.elapsed + dt) );
}