#include "planner.h"

using namespace Capt;

Planner::Planner(Model *model, Param *param, Config *config, Grid *grid, Capturability *capt){
  tree      = new Tree(grid, capt);
  search    = new Search(grid, tree);
  swingFoot = new SwingFoot(model);
  pendulum  = new Pendulum(model);

  config->read(&dt, "timestep");
  config->read(&preview, "preview");
}

Planner::~Planner(){
  delete tree;
  delete search;
}

void Planner::set(planner::Input input){
  this->input = input;
}

planner::Output Planner::get(){
  return this->output;
}

std::vector<Sequence> Planner::getSequence(){
  return search->getSequence();
}

arr3_t Planner::getFootstepR(){
  return search->getFootstepR();
}

arr3_t Planner::getFootstepL(){
  return search->getFootstepL();
}

void Planner::plan(){
  rfoot = vec3Tovec2(input.rfoot);
  lfoot = vec3Tovec2(input.lfoot);
  icp   = vec3Tovec2(input.icp);
  s_suf = input.s_suf;

  calculateGoal();
  runSearch();
}

void Planner::replan(){
  // calculateStart();
  calculateGoal();
  runSearch();
}

void Planner::calculateStart(){

}

void Planner::calculateGoal(){
  vec3_t suf;
  if(input.s_suf == FOOT_R) {
    suf = input.rfoot;
  }else{
    suf = input.lfoot;
  }

  double distMin       = 100; // set very large value as initial value
  int    currentFootId = 0;
  for(size_t i = 0; i < input.footstep.size(); i++) {
    if(input.footstep[i].suf == input.s_suf) {
      double dist = ( input.footstep[i].pos - suf ).norm();
      if(dist < distMin) {
        distMin       = dist;
        currentFootId = (int)i;
      }
    }
  }
  g_suf = input.footstep[currentFootId + preview].suf;
  vec3_t g_foot = input.footstep[currentFootId + preview].pos;
  goal = vec3Tovec2(g_foot);
}

void Planner::runSearch(){
  search->setStart(rfoot, lfoot, icp, s_suf);
  search->setGoal(goal, g_suf);
  found = search->calc();

  if(found) { // if found solution
    // get state & input
    State s = search->getState();
    Input i = search->getInput();

    // calc swing foot trajectory
    swingFoot->set(s.swf, i.swf);

    // calculate distance
    double de = 0.0, dr = 0.0;
    if(s_suf == FOOT_R) {
      de         = ( s.swf - lfoot ).norm();
      dr         = ( i.swf - lfoot ).norm();
      output.suf = vec2Tovec3(rfoot);
    }else{
      de         = ( s.swf - rfoot ).norm();
      dr         = ( i.swf - rfoot ).norm();
      output.suf = vec2Tovec3(lfoot);
    }

    // set to output
    output.computation = 0.0;
    output.duration    = swingFoot->getTime();
    output.alpha       = de / ( de + dr );
    output.cop         = vec2Tovec3(i.cop);
    output.icp         = vec2Tovec3(s.icp);
    output.swf         = vec2Tovec3(s.swf);
    output.land        = vec2Tovec3(i.swf);
  }else{ // couldn't found solution or reached goal
  }
}

std::vector<CaptData> Planner::getCaptureRegion(){
  return search->getCaptureRegion();
}