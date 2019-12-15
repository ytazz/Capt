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

  if(input.elapsed < dt) {
    calculateGoal();
    runSearch();
  }
}

planner::Output Planner::get(double time){
  generatePath(time);
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

double Planner::getPreviewRadius(){
  return preview;
}

void Planner::calculateGoal(){
  vec3_t suf;
  if(input.s_suf == FOOT_R) {
    suf = input.rfoot;
  }else{
    suf = input.lfoot;
  }

  printf("%lf\n", preview);
  for(size_t i = 0; i < input.footstep.size(); i++) {
    double dist = ( input.footstep[i].pos - suf ).norm();
    printf("%1.3lf\n", dist);
    if(dist <= preview) {
      g_suf  = input.footstep[i].suf;
      g_foot = input.footstep[i].pos;
    }
  }
  printf("goal\n");
  if(g_suf == FOOT_R) {
    printf("suf: FOOT_R\n");
  }else{
    printf("suf: FOOT_L\n");
  }
  printf("g_foot %1.3lf, %1.3lf\n", g_foot.x(), g_foot.y() );

  // g_foot.x() = 0.75;
  // g_foot.y() = 0.05;
  // g_suf      = FOOT_R;
}

void Planner::runSearch(){
  vec2_t rfoot = vec3Tovec2(input.rfoot);
  vec2_t lfoot = vec3Tovec2(input.lfoot);
  vec2_t icp   = vec3Tovec2(input.icp);
  vec2_t goal  = vec3Tovec2(g_foot);
  search->setStart(rfoot, lfoot, icp, input.s_suf);
  search->setGoal(goal, g_suf);
  found = search->calc();

  if(found) { // if found solution
    // get state & input
    State s = search->getState();
    Input i = search->getInput();

    // calc swing foot trajectory
    swingFoot->set(s.swf, i.swf);

    // calc icp trajectory
    pendulum->setCop(i.cop);
    pendulum->setIcp(s.icp);

    // set to output
    output.duration = swingFoot->getTime();
  }else{ // couldn't found solution or reached goal
  }
}

void Planner::generatePath(double time){
  if(found) {
    if(input.s_suf == FOOT_R) {
      output.rfoot = input.rfoot;
      output.lfoot = swingFoot->getTraj(time);
    }else{
      output.rfoot = swingFoot->getTraj(time);
      output.lfoot = input.lfoot;
    }
    output.icp = vec2Tovec3(pendulum->getIcp(time) );
    output.cop = vec2Tovec3(pendulum->getCop(time) );
  }else{
    output.cop = output.icp;
  }
}

std::vector<CaptData> Planner::getCaptureRegion(){
  return search->getCaptureRegion();
}