#include "capturability.h"

namespace Capt {

Capturability::Capturability(Model model, Param param)
  : grid(param), model(model) {
  // this->model = model;
  zero_data = new int [grid.getNumState()];
  n_data    = new CaptureSet*[grid.getNumState()];
  for (int i = 0; i < grid.getNumState(); i++) {
    n_data[i] = new CaptureSet[grid.getNumInput()];
  }

  foot_vel      = model.getVal("physics", "foot_vel_max");
  step_time_min = model.getVal("physics", "step_time_min");
}

Capturability::~Capturability() {
}

void Capturability::load(std::string file_name, DataType type) {
  FILE *fp;
  int   num_data = 0;

  if (type == ZERO_STEP) {
    int        buf[2];
    CaptureSet set;

    if ( ( fp = fopen(file_name.c_str(), "r") ) == NULL) {
      printf("Error: Couldn't find the file \"%s\"\n", file_name.c_str() );
      exit(EXIT_FAILURE);
    } else {
      printf("Find 0-step database.\n");
      while (fscanf(fp, "%d, %d", &buf[0], &buf[1]) != EOF) {
        zero_data[buf[0]] = buf[1];
        num_data++;
      }
      fclose(fp);
    }
  } else if (type == N_STEP) {
    // make cop list
    vec2_t             *cop = (vec2_t*)malloc(sizeof( vec2_t ) * grid.getNumState() );
    Polygon             polygon;
    std::vector<vec2_t> region = model.getVec("foot", "foot_r_convex");
    for(int state_id = 0; state_id < grid.getNumState(); state_id++) {
      State state = grid.getState(state_id);
      cop[state_id] = polygon.getClosestPoint(state.icp, region);
    }

    int        buf[4];
    CaptureSet set;

    if ( ( fp = fopen(file_name.c_str(), "r") ) == NULL) {
      printf("Error: Couldn't find the file \"%s\"\n", file_name.c_str() );
      exit(EXIT_FAILURE);
    } else {
      printf("Find N-step database.\n");
      while (fscanf(fp, "%d,%d,%d,%d", &buf[0], &buf[1], &buf[2], &buf[3]) != EOF) {
        if(buf[0] == 367846) {
          set.state_id      = buf[0];
          set.input_id      = buf[1];
          set.next_state_id = buf[2];
          set.n_step        = buf[3];

          State state = grid.getState(set.state_id);
          Input input = grid.getInput(set.input_id);

          set.swft      = input.swft;
          set.cop       = cop[set.state_id];
          set.step_time = ( input.swft - state.swft ).norm() / foot_vel + step_time_min;

          n_data[set.state_id][set.input_id] = set;
          num_data++;
        }
      }
      fclose(fp);
    }
  }

  printf("Read success! (%d datas)\n", num_data);
}

void Capturability::setCaptureSet(const int state_id, const int input_id,
                                  const int next_state_id, const int n_step,
                                  const vec2_t cop, const float step_time) {
  CaptureSet set;
  set.state_id      = state_id;
  set.input_id      = input_id;
  set.next_state_id = next_state_id;
  set.n_step        = n_step;
  set.swft          = grid.getInput(set.input_id).swft;
  set.cop           = cop;
  set.step_time     = step_time;

  // n_data.push_back(set);
}

void Capturability::setCaptureSet(const int state_id, const int input_id,
                                  const int next_state_id, const int n_step) {
  vec2_t v;
  v.clear();

  CaptureSet set;
  set.state_id      = state_id;
  set.input_id      = input_id;
  set.next_state_id = next_state_id;
  set.n_step        = n_step;
  set.swft          = grid.getInput(set.input_id).swft;
  set.cop           = v;
  set.step_time     = 0.0;

  // n_data.push_back(set);
}

std::vector<CaptureSet> Capturability::getCaptureRegion(const int state_id,
                                                        const int n_step) {

  std::vector<CaptureSet> sets;

  sets.clear();
  for (int i = 0; i < grid.getNumInput(); i++) {
    if (n_data[state_id][i].n_step == n_step) {
      sets.push_back(n_data[state_id][i]);
    }
  }

  return sets;
}

std::vector<CaptureSet> Capturability::getCaptureRegion(const State state,
                                                        const int n_step) {

  std::vector<CaptureSet> sets;

  if (grid.existState(state) ) {
    sets = getCaptureRegion(grid.getStateIndex(state), n_step);
  }

  return sets;
}

bool Capturability::capturable(State state, int n_step) {
  bool flag = false;

  int state_id = grid.getStateIndex(state);

  if (n_step == 0) {
    if (zero_data[state_id] == 0)
      flag = true;
  } else {
    if (!getCaptureRegion(state, n_step).empty() )
      flag = true;
  }

  return flag;
}

bool Capturability::capturable(int state_id, int n_step) {
  bool flag = false;

  State state_ = grid.getState(state_id);
  flag = capturable(state_, n_step);

  return flag;
}

void Capturability::save(const char *file_name, const int n_step) {
  FILE *fp = fopen(file_name, "w");
  // fprintf(fp, "state_id,input_id,next_state_id,n_step,cop_x,cop_y,time\n");
  // for (size_t i = 0; i < n_data.size(); i++) {
  //   if (n_data[i].n_step == n_step) {
  //     fprintf(fp, "%d,", n_data[i].state_id);
  //     fprintf(fp, "%d,", n_data[i].input_id);
  //     fprintf(fp, "%d,", n_data[i].next_state_id);
  //     fprintf(fp, "%d,", n_data[i].n_step);
  //     fprintf(fp, "%f,%f,", n_data[i].cop.x, n_data[i].cop.y);
  //     fprintf(fp, "%f,", n_data[i].step_time);
  //     fprintf(fp, "\n");
  //   }
  // }
  fclose(fp);
}

} // namespace Capt