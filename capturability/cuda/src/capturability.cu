#include "capturability.cuh"

namespace GPGPU {

__device__ Capturability::Capturability(Model model, Param param)
    : grid(param), model(model) {
  // this->model = model;
}

__device__ Capturability::~Capturability() {}

__device__ void Capturability::load(const char *file_name) {
  FILE *fp;
  int num_data = 0;
  int ibuf[4];
  float fbuf[3];
  CaptureSet set;

  if ((fp = fopen(file_name, "r")) == NULL) {
    printf("Error: Couldn't find the file \"%s\"\n", file_name);
    exit(EXIT_FAILURE);
  } else {
    while (fscanf(fp, "%d,%d,%d,%d,%f,%f,%f", &ibuf[0], &ibuf[1], &ibuf[2],
                  &ibuf[3], &fbuf[0], &fbuf[1], &fbuf[2]) != EOF) {
      set.state_id = ibuf[0];
      set.input_id = ibuf[1];
      set.next_state_id = ibuf[2];
      set.n_step = ibuf[3];
      set.swft = grid.getInput(set.input_id).swft;
      set.cop.setCartesian(fbuf[0], fbuf[1]);
      set.step_time = fbuf[2];
      capture_set.push_back(set);
      num_data++;
    }
    fclose(fp);
  }

  printf("Read success! (%d datas)\n", num_data);
}

__device__ void Capturability::setCaptureSet(const int state_id,
                                             const int input_id,
                                             const int next_state_id,
                                             const int n_step, const vec2_t cop,
                                             const float step_time) {
  CaptureSet set;
  set.state_id = state_id;
  set.input_id = input_id;
  set.next_state_id = next_state_id;
  set.n_step = n_step;
  set.swft = grid.getInput(set.input_id).swft;
  set.cop = cop;
  set.step_time = step_time;

  capture_set.push_back(set);
}

__device__ void Capturability::setCaptureSet(const int state_id,
                                             const int input_id,
                                             const int next_state_id,
                                             const int n_step) {
  vec2_t v;
  v.clear();

  CaptureSet set;
  set.state_id = state_id;
  set.input_id = input_id;
  set.next_state_id = next_state_id;
  set.n_step = n_step;
  set.swft = grid.getInput(set.input_id).swft;
  set.cop = v;
  set.step_time = 0.0;

  capture_set.push_back(set);
}

__device__ std::vector<CaptureSet>
Capturability::getCaptureRegion(const int state_id, const int n_step) {

  std::vector<CaptureSet> sets;

  sets.clear();
  for (size_t i = 0; i < capture_set.size(); i++) {
    if (capture_set[i].state_id == state_id &&
        capture_set[i].n_step == n_step) {
      sets.push_back(capture_set[i]);
    }
  }

  return sets;
}

__device__ std::vector<CaptureSet>
Capturability::getCaptureRegion(const State state, const int n_step) {

  std::vector<CaptureSet> sets;

  if (grid.existState(state)) {
    sets = getCaptureRegion(grid.getStateIndex(state), n_step);
  }

  return sets;
}

__device__ bool Capturability::capturable(State state, int n_step) {
  bool flag = false;

  if (n_step == 0) {
    Polygon polygon;

    // only SSP
    std::vector<vec2_t> region = model.getVec("foot", "foot_r_convex");
    flag = polygon.inPolygon(state.icp, region);

    // considered DSP
    // std::vector<vec2_t> region_r = model.getVec("foot", "foot_r");
    // std::vector<vec2_t> region_l = model.getVec("foot", "foot_l",
    // state.swft);
    // polygon.setVertex(region_r);
    // polygon.setVertex(region_l);
    // flag = polygon.inPolygon(state.icp, polygon.getConvexHull());

  } else {
    if (!getCaptureRegion(state, n_step).empty())
      flag = true;
  }

  return flag;
}

__device__ bool Capturability::capturable(int state_id, int n_step) {
  bool flag = false;

  State state_ = grid.getState(state_id);
  flag = capturable(state_, n_step);

  return flag;
}

__device__ void Capturability::save(const char *file_name, const int n_step) {
  FILE *fp = fopen(file_name, "w");
  // fprintf(fp, "state_id,input_id,next_state_id,n_step,cop_x,cop_y,time\n");
  for (size_t i = 0; i < capture_set.size(); i++) {
    if (capture_set[i].n_step == n_step) {
      fprintf(fp, "%d,", capture_set[i].state_id);
      fprintf(fp, "%d,", capture_set[i].input_id);
      fprintf(fp, "%d,", capture_set[i].next_state_id);
      fprintf(fp, "%d,", capture_set[i].n_step);
      fprintf(fp, "%f,%f,", capture_set[i].cop.x, capture_set[i].cop.y);
      fprintf(fp, "%f,", capture_set[i].step_time);
      fprintf(fp, "\n");
    }
  }
  fclose(fp);
}

} // namespace GPGPU