#include "analysis_cpu.h"

namespace Capt {

const int nmax = 10;

Analysis::Analysis(Model *model, Param *param, Grid *grid)
  : model(model), param(param), grid(grid)
{
  model->read(&v_max, "foot_vel_max");
  param->read(&z_max, "swf_z_max");

  model->read(&g, "gravity");
  model->read(&h, "com_height");
  T = sqrt(h/g);

  printf("*** Analysis ***\n");
  printf("  Initializing ... ");
  fflush(stdout);
  initState();
  printf("Done!\n");

  printf("  Calculating .... ");
  fflush(stdout);
  calcBasin();
  printf("Done!\n");
}

Analysis::~Analysis() {
}

void Analysis::initState(){
  int state_num = grid->getNumState();
  state.resize(state_num);
  for (int state_id = 0; state_id < state_num; state_id++)
    state[state_id] = grid->getState(state_id);
}

void Analysis::calcBasin(){
  arr2_t foot_r;
  arr2_t foot_l;
  arr2_t region;
  model->read(&foot_r, "foot_r_convex");

  /*
  if(enableDoubleSupport) {
    for(int state_id = 0; state_id < state_num; state_id++) {
      model->read(&foot_l, "foot_l_convex", vec3Tovec2(state[state_id].swf) );
      Polygon polygon;
      polygon.setVertex(foot_r);
      polygon.setVertex(foot_l);
      region = polygon.getConvexHull();
      if(grid->isSteppable(vec3Tovec2(state[state_id].swf) ) ) {
        if(polygon.inPolygon(state[state_id].icp, region) && state[state_id].swf.z() < EPSILON )
          basin[state_id] = 0;
      }
    }
  }
  else{
  */
  // enumerate all valid states
  int state_num = grid->getNumState();
  state_id_remain.clear();
  for(int state_id = 0; state_id < state_num; state_id++) {
    State& st = state[state_id];
    if( grid->isSteppable(vec3Tovec2(st.swf)) )
      state_id_remain.push_back(state_id);
  }

  // calculate 0-step capture basin
  Polygon polygon;
  state_id_remain_next.clear();
  for(int state_id : state_id_remain) {
    State& st = state[state_id];
    if( polygon.inPolygon(st.icp, foot_r) &&   //< icp is inside support polygon
        st.swf.z() < EPSILON ) {               //< swing foot is on the ground
        cap_basin[0].push_back(state_id);
        cap_region.push_back(CaptureRegion(state_id, -1, 0));
    }
    else{
      state_id_remain_next.push_back(state_id);
    }
  }
  state_id_remain.swap(state_id_remain_next);
}

/*
void Analysis::calcTrans(){
  Pendulum pendulum(model);
  vec2_t   icp;

  int state_num = grid->getNumState();
  int input_num = grid->getNumInput();

  int num = 0;

  for(int state_id = 0; state_id < state_num; state_id++)
  for(int input_id = 0; input_id < input_num; input_id++) {
    int id = state_id * input_num + input_id;
    trans[id] = -1;

    State& st = state[state_id];
    Input& in = input[input_id];

    if( grid->isSteppable(vec3Tovec2(st.swf)) &&
        grid->isSteppable(in.swf) ) {

      pendulum.setIcp(st.icp);
      pendulum.setCop(in.cop);
      swing->set(st.swf, vec2Tovec3(in.swf) );
      icp = pendulum.getIcp(swing->getDuration() );

      State state_;
      state_.icp = vec2_t(-in.swf.x() + icp.x(), in.swf.y() - icp.y());
      state_.swf = vec3_t(-in.swf.x(), in.swf.y(), 0);

      trans[id] = grid->roundState(state_).id;
      if(trans[id] != -1)
        num++;
    }
  }

  printf("num grid: %d  num valid: %d  percentage: %f\n", state_num*input_num, num, (double)num/(double)(state_num*input_num));
}
*/

bool Analysis::exe(int n){
  if(n == 0)
    return false;

  Pendulum pendulum(model);
  vec2_t icp;

  bool found_at_all = false;
  state_id_remain_next.clear();
  for(int state_id : state_id_remain){
    State& st = state[state_id];

    bool found = false;

    // enumerate states in (N-1)-step capture basin
    for(int next_id : cap_basin[n-1]){
      State& stnext = state[next_id];

      // calculate input
      Input in = calcInput(st, stnext);

      // check feasibility of input
      if(!grid->isSteppable(in.swf))
        continue;
      if(!grid->isValidCop(in.cop))
        continue;

      if(!found){
        found        = true;
        found_at_all = true;
        cap_basin[n].push_back(state_id);
      }
      cap_region.push_back(CaptureRegion(state_id, next_id, n));
    }
    if(!found)
      state_id_remain_next.push_back(state_id);

    /*
    // test possible inputs
    for(int input_id = 0; input_id < input_num; input_id++){
      Input& in = input[input_id];
      // skip if out of steppable region
      if(!grid->isSteppable(in.swf))
        continue;

      // calc next state
      pendulum.setIcp(st.icp);
      pendulum.setCop(in.cop);
      swing->set(st.swf, vec2Tovec3(in.swf) );
      icp = pendulum.getIcp(swing->getDuration() );

      State state_;
      state_.icp = vec2_t(-in.swf.x() + icp.x(), in.swf.y() - icp.y());
      state_.swf = vec3_t(-in.swf.x(), in.swf.y(), 0);
      int next_state_id = grid->roundState(state_).id;
      if(next_state_id == -1)
        continue;

      // if next state is N-1 step basin then mark this state as N-step basin
      if(basin[next_state_id] == n-1){
        basin[state_id] = n;
        nstep[n].push_back(Tuple(state_id, input_id, next_state_id));
        found = true;
      }
    }
    */
  }
    /* old version that uses precomputed trans
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
    */
  return found_at_all;
}

void Analysis::exe(){
  printf("  Analysing ...... ");
  fflush(stdout);

  int n = 1;
  while(n <= nmax){
    if(!exe(n))
      break;
    printf(" %d", n);
    n++;
  }

  printf("Done!\n");
}

void Analysis::save(std::string file_name, bool header, bool binary){
  FILE *fp = fopen(file_name.c_str(), binary ? "wb" : "w");

  // Header
  if (header && !binary) {
    fprintf(fp, "%s,", "state_id");
    fprintf(fp, "%s,", "next_id");
    fprintf(fp, "%s", "nstep");
    fprintf(fp, "\n");
  }

  // Data
  for(CaptureRegion& r : cap_region){
    if(binary){
      fwrite(&r.state_id, sizeof(int), 1, fp);
      fwrite(&r.next_id , sizeof(int), 1, fp);
      fwrite(&r.nstep   , sizeof(int), 1, fp);
    }
    else{
      fprintf(fp, "%d,%d,%d\n", r.state_id, r.next_id, r.nstep);
    }
  }

  fclose(fp);
}

} // namespace Capt