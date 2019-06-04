#include "capturability.h"

namespace CA {

Capturability::Capturability(Model model, Param param)
    : grid(param), model(model) {
  // this->model = model;
}

Capturability::~Capturability() {}

void Capturability::load(const char *file_name) {
  FILE *fp;
  int num_data = 0;
  float fbuf[6];
  int ibuf[3];

  if ((fp = fopen(file_name, "r")) == NULL) {
    printf("Error: Couldn't find the file \"%s\"\n", file_name);
    exit(EXIT_FAILURE);
  } else {
    while (fscanf(fp, "%d,%f,%f,%f,%f,%d,%d,%f,%f", &ibuf[0], &fbuf[0],
                  &fbuf[1], &fbuf[2], &fbuf[3], &ibuf[1], &ibuf[2], &fbuf[4],
                  &fbuf[5]) != EOF) {
      GridState grid_state;
      GridInput grid_input;
      // GridState next_grid_state;

      grid_state.id = ibuf[0];
      grid_state.state = grid.getState(grid_state.id);

      grid_input.id = ibuf[2];
      grid_input.input = grid.getInput(grid_input.id);

      // next_grid_state.id = buf[8];
      // next_grid_state.state = grid.getState(next_grid_state.id);
      //
      // setCaptureSet(grid_state, grid_input, next_grid_state, 1);
      num_data++;
    }
    fclose(fp);
  }

  printf("Read success! (%d datas)\n", num_data);
}

void Capturability::setCaptureSet(const int state_id, const int input_id,
                                  const int next_state_id,
                                  const int n_step_capturable) {
  CaptureSet set;
  set.state_id = state_id;
  set.input_id = input_id;
  set.n = n_step_capturable;

  capture_set.push_back(set);
}

bool Capturability::capturable(State state, int n_step) {
  bool flag = false;

  if (n_step == 0) {
    Polygon polygon;
    polygon.setVertex(model.getVec("link", "foot_r"));
    flag = polygon.inPolygon(state.icp, polygon.getConvexHull());
  } else {
  }

  return flag;
}

void Capturability::save(const char *file_name, const int n_step_capturable) {
  FILE *fp = fopen(file_name, "w");
  fprintf(fp, "state_id,input_id,next_state_id,n_step\n");
  for (size_t i = 0; i < capture_set.size(); i++) {
    if (capture_set[i].n == n_step_capturable) {
      fprintf(fp, "%d,", capture_set[i].state_id);
      fprintf(fp, "%d,", capture_set[i].input_id);
      fprintf(fp, "%d,", capture_set[i].next_state_id);
      fprintf(fp, "%d,", capture_set[i].n);
      fprintf(fp, "\n");
    }
  }
}

} // namespace CA