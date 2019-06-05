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
  int buf[4];
  CaptureSet set;

  if ((fp = fopen(file_name, "r")) == NULL) {
    printf("Error: Couldn't find the file \"%s\"\n", file_name);
    exit(EXIT_FAILURE);
  } else {
    while (fscanf(fp, "%d,%d,%d,%d", &buf[0], &buf[1], &buf[2], &buf[3]) !=
           EOF) {
      set.state_id = buf[0];
      set.input_id = buf[1];
      set.next_state_id = buf[2];
      set.n = buf[3];
      capture_set.push_back(set);
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
  set.next_state_id = next_state_id;
  set.n = n_step_capturable;

  capture_set.push_back(set);
}

std::vector<Input>
Capturability::getCaptureRegion(const int state_id,
                                const int n_step_capturable) {

  std::vector<Input> capture_region;
  Input capture_point;

  capture_region.clear();
  for (size_t i = 0; i < capture_set.size(); i++) {
    if (capture_set[i].state_id == state_id &&
        capture_set[i].n == n_step_capturable) {
      capture_point = grid.getInput(capture_set[i].input_id);
      capture_region.push_back(capture_point);
    }
  }

  return capture_region;
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