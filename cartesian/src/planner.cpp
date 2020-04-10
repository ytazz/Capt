#include "planner.h"

using namespace Capt;

#if 0
Planner::Planner(Model *model, Param *param, Config *config, Grid *grid, Capturability *capt){
  tree     = new Tree(grid, capt);
  search   = new Search(grid, tree);
  swing    = new Swing(model, param);
  pendulum = new Pendulum(model);

  model->read(&dt_min, "step_time_min");

  config->read(&dt, "timestep");
  config->read(&preview, "preview");
}

Planner::~Planner(){
  delete tree;
  delete search;
}

void Planner::set(EnhancedState state){
  this->state = state;
}

EnhancedInput Planner::get(){
  return this->input;
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

void Planner::clear(){
  search->clear();
}

Status Planner::plan(){
  rfoot = state.rfoot;
  lfoot = state.lfoot;
  icp   = state.icp;
  s_suf = state.s_suf;

  if(state.s_suf == FOOT_R) {
    suf = rfoot;
    swf = lfoot;
  }else{
    suf = lfoot;
    swf = rfoot;
  }

  Status status;
  if(calculateGoal() ) {
    status = runSearch(preview);
  }else{
    status = Status::FINISH;
  }

  return status;
}

void Planner::calculateStart(){

}

bool Planner::calculateGoal(){
  float  distMin       = 100; // set very large value as initial value
  int    currentFootId = 0;
  int    maxFootId     = (int)state.footstep.size() - 1;
  for(int i = 0; i <= maxFootId; i++) {
    if(state.footstep[i].suf == state.s_suf) {
      float dist = ( state.footstep[i].pos - suf ).norm();
      if(dist < distMin) {
        distMin       = dist;
        currentFootId = (int)i;
      }
    }
  }
  if(currentFootId == maxFootId ) {
    return false;
  }

  int remainedFootsteps = maxFootId - currentFootId;
  if(preview <= remainedFootsteps) {
    posRef.clear();
    icpRef.clear();
    posRef.resize(preview);
    icpRef.resize(preview);
    for(int i = 0; i < preview; i++) {
      posRef[i] = vec3Tovec2(state.footstep[currentFootId + i].pos);
      icpRef[i] = vec3Tovec2(state.footstep[currentFootId + i].icp);
      // printf("posRef: %+1.3lf %+1.3lf\n", posRef[i].x(), posRef[i].y() );
    }
  }else{
    posRef.clear();
    icpRef.clear();
    posRef.resize(remainedFootsteps + 1);
    icpRef.resize(remainedFootsteps + 1);
    for(int i = 0; i <= remainedFootsteps; i++) {
      posRef[i] = vec3Tovec2(state.footstep[currentFootId + i].pos);
      icpRef[i] = vec3Tovec2(state.footstep[currentFootId + i].icp);
      // printf("posRef: %+1.3lf %+1.3lf\n", posRef[i].x(), posRef[i].y() );
    }
  }

  return true;
}

Status Planner::runSearch(int preview){
  search->setStart(rfoot, lfoot, icp, s_suf);
  search->setReference(posRef, icpRef);
  found = search->calc( (int)posRef.size() - 1);

  Status status;
  if(found) { // if found solution
    // get state & input
    State s = search->getState();
    Input i = search->getInput();

    // calc swing foot trajectory
    swing->set(swf, vec2Tovec3(i.swf) );

    // set to output
    input.duration = swing->getDuration();
    input.suf      = suf;
    input.swf      = swf;
    input.cop      = vec2Tovec3(i.cop);
    input.icp      = vec2Tovec3(s.icp);
    input.land     = vec2Tovec3(i.swf);
    // printf("duration  %1.3lf\n", input.duration );
    // printf("swf  %1.3lf, %1.3lf\n", input.swf.x(), input.swf.y() );
    // printf("land %1.3lf, %1.3lf\n", input.land.x(), input.land.y() );

    status = Status::SUCCESS;
  }else{ // couldn't found solution or reached goal
    status = Status::FAIL;
  }

  return status;
}

std::vector<CaptData> Planner::getCaptureRegion(){
  return search->getCaptureRegion();
}
#endif