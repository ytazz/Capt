#include "analysis_cpu.h"

namespace Capt {

Analysis::Analysis(Model *model, Grid *grid)
  : model(model), grid(grid),
    state_num(grid->getNumState() ), input_num(grid->getNumInput() ), grid_num(grid->getNumGrid() ),
    epsilon(0.001){
  model->read(&foot_vel_max, "foot_vel_max");
  model->read(&step_time_min, "step_time_min");

  printf("*** Analysis ***\n");
  printf("  Initializing ... ");
  fflush(stdout);
  initState();
  initInput();
  initTrans();
  initBasin();
  initNstep();
  printf("Done!\n");

  printf("  Calculating .... ");
  fflush(stdout);
  calcBasin();
  calcTrans();
  printf("Done!\n");
}

Analysis::~Analysis() {
}

void Analysis::initState(){
  state = new State[state_num];
  for (int state_id = 0; state_id < state_num; state_id++)
    state[state_id] = grid->getState(state_id);
}

void Analysis::initInput(){
  input = new Input[input_num];
  for (int input_id = 0; input_id < input_num; input_id++)
    input[input_id] = grid->getInput(input_id);
}

void Analysis::initTrans(){
  trans = new int[grid_num];
  for (int id = 0; id < grid_num; id++)
    trans[id] = -1;
}

void Analysis::initBasin(){
  basin = new int[state_num];
  for (int state_id = 0; state_id < state_num; state_id++)
    basin[state_id] = -1;
}

void Analysis::initNstep(){
  nstep = new int[grid_num];
  for (int id = 0; id < grid_num; id++)
    nstep[id] = -1;
}

void Analysis::calcBasin(){
  arr2_t foot_r;
  arr2_t foot_l;
  arr2_t region;
  model->read(&foot_r, "foot_r_convex");

  if(enableDoubleSupport) {
    for(int state_id = 0; state_id < state_num; state_id++) {
      model->read(&foot_l, "foot_l_convex", state[state_id].swf);
      Polygon polygon;
      polygon.setVertex(foot_r);
      polygon.setVertex(foot_l);
      region = polygon.getConvexHull();
      if(polygon.inPolygon(state[state_id].icp, region) )
        basin[state_id] = 0;
    }
  }else{
    for(int state_id = 0; state_id < state_num; state_id++) {
      Polygon polygon;
      if(polygon.inPolygon(state[state_id].icp, foot_r) && state[state_id].elp < epsilon )
        basin[state_id] = 0;
    }
  }
}

void Analysis::calcTrans(){
  Pendulum pendulum(model);
  vec2_t   icp;

  FILE *fp = fopen("stepTime.csv", "w");

  fprintf(fp, "state_id, input_id, trans_id, tau, icp.x, icp.y\n");
  for(int state_id = 0; state_id < state_num; state_id++) {
    for(int input_id = 0; input_id < input_num; input_id++) {
      int id = state_id * input_num + input_id;

      double dist = ( input[input_id].swf - state[state_id].swf ).norm();
      double tau  = max(0, step_time_min / 2 - state[state_id].elp) +
                    dist / foot_vel_max + step_time_min / 2;

      pendulum.setIcp(state[state_id].icp);
      pendulum.setCop(input[input_id].cop);
      icp = pendulum.getIcp(tau);

      State state_;
      state_.icp << -input[input_id].swf.x() + icp.x(), input[input_id].swf.y() - icp.y();
      state_.swf << -input[input_id].swf.x(), input[input_id].swf.y();
      state_.elp = 0.0;

      trans[id] = grid->roundState(state_).id;
      if(trans[id] >= 0)
        fprintf(fp, "%d, %d, %d, %lf, %lf, %lf\n", state_id, input_id, trans[id], tau, state_.icp.x(), state_.icp.y() );
    }
  }

  fclose(fp);
}

bool Analysis::exe(const int n){
  bool found = false;
  if(n > 0) {
    for(int state_id = 0; state_id < state_num; state_id++) {
      for(int input_id = 0; input_id < input_num; input_id++) {
        int id = state_id * input_num + input_id;
        if (trans[id] >= 0) {
          if (basin[trans[id]] == ( n - 1 ) ) {
            nstep[id] = n;
            found     = true;
            if (basin[state_id] < 0) {
              basin[state_id] = n;
            }
          }
        }
      }
    }
  }

  return found;
}

void Analysis::exe(){
  printf("  Analysing ...... ");
  fflush(stdout);

  max_step = 1;
  while(exe(max_step) ) {
    max_step++;
  }
  max_step--;

  printf("Done!\n");
}

void Analysis::saveBasin(std::string file_name, bool header){
  FILE *fp = fopen(file_name.c_str(), "w");

  // Header
  if (header) {
    // fprintf(fp, "%s,", "state_id");
    fprintf(fp, "%s", "nstep");
    fprintf(fp, "\n");
  }

  // Data
  for (int state_id = 0; state_id < state_num; state_id++) {
    // fprintf(fp, "%d,", state_id);
    fprintf(fp, "%d", basin[state_id]);
    fprintf(fp, "\n");
  }

  fclose(fp);
}

void Analysis::saveNstep(std::string file_name, bool header){
  FILE *fp = fopen(file_name.c_str(), "w");

  // Header
  if (header) {
    // fprintf(fp, "%s,", "state_id");
    // fprintf(fp, "%s,", "input_id");
    fprintf(fp, "%s,", "trans");
    fprintf(fp, "%s", "nstep");
    fprintf(fp, "\n");
  }

  // Data
  int num_step[max_step + 1];
  for(int i = 0; i < max_step + 1; i++) {
    num_step[i] = 0;
  }
  for (int state_id = 0; state_id < state_num; state_id++) {
    for (int input_id = 0; input_id < input_num; input_id++) {
      int id = state_id * input_num + input_id;
      // fprintf(fp, "%d,", state_id);
      // fprintf(fp, "%d,", input_id);
      fprintf(fp, "%d,", trans[id]);
      fprintf(fp, "%d", nstep[id]);
      fprintf(fp, "\n");

      if(nstep[id] > 0)
        num_step[nstep[id]]++;
    }
  }

  printf("*** Result ***\n");
  printf("  Feasible maximum steps: %d\n", max_step);
  for(int i = 1; i <= max_step; i++) {
    printf("  %d-step capture point  : %8d\n", i, num_step[i]);
  }

  fclose(fp);
}

} // namespace Capt