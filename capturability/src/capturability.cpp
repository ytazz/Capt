#include "capturability.h"

namespace Capt {

Capturability::Capturability(Grid *grid)
  : grid(grid) {
  state_num = grid->getNumState();
  input_num = grid->getNumInput();

  data_basin = new int [state_num];
  data_nstep = new CaptureSet*[state_num];
  for (int i = 0; i < state_num; i++) {
    data_nstep[i] = new CaptureSet[input_num];
  }

  max_step = 0;
}

// Capturability::Capturability(const Capturability &obj){
//   data_basin = new int [state_num];
//   data_nstep = new CaptureSet*[state_num];
//   for (int i = 0; i < state_num; i++) {
//     data_nstep[i] = new CaptureSet[input_num];
//   }
// }

Capturability::~Capturability() {
}

void Capturability::load(std::string file_name, DataType type) {
  FILE *fp = fopen(file_name.c_str(), "r");
  if ( fp == NULL) {
    printf("Error: Couldn't find the file \"%s\"\n", file_name.c_str() );
    exit(EXIT_FAILURE);
  }
  int id = 0;

  if (type == BASIN) {
    printf("Find Basin database.\n");

    int buf;
    while (fscanf(fp, "%d", &buf) != EOF) {
      data_basin[id] = buf;
      id++;
    }
  } else if (type == NSTEP) {
    printf("Find N-step database.\n");

    int buf[2];
    while (fscanf(fp, "%d,%d", &buf[0], &buf[1]) != EOF) {
      int state_id = id / input_num;
      int input_id = id % input_num;

      CaptureSet* set = getCaptureSet(state_id, input_id);
      set->state_id = state_id;
      set->input_id = input_id;
      set->next_id  = buf[0];
      set->nstep    = buf[1];

      if(max_step < set->nstep)
        max_step = set->nstep;

      id++;
    }
  } else if (type == STEPTIME) {
    printf("Find Step-Time database.\n");

    double buf;
    while (fscanf(fp, "%lf", &buf) != EOF) {
      int state_id = id / input_num;
      int input_id = id % input_num;

      CaptureSet* set = getCaptureSet(state_id, input_id);
      set->step_time = buf;

      id++;
    }
  }

  printf("Read success! (%d datas)\n", id);
  fclose(fp);
}

CaptureSet* Capturability::getCaptureSet(int state_id, int input_id){
  return &data_nstep[state_id][input_id];
}

std::vector<CaptureSet> Capturability::getCaptureRegion(const int state_id) {
  std::vector<CaptureSet> sets;

  sets.clear();
  for (int i = 0; i < grid->getNumInput(); i++) {
    if (data_nstep[state_id][i].nstep > 0) {
      sets.push_back(data_nstep[state_id][i]);
    }
  }

  return sets;
}

std::vector<CaptureSet> Capturability::getCaptureRegion(const int state_id, const int nstep) {
  std::vector<CaptureSet> sets;

  sets.clear();
  for (int i = 0; i < grid->getNumInput(); i++) {
    if (data_nstep[state_id][i].nstep == nstep) {
      sets.push_back(data_nstep[state_id][i]);
    }
  }

  return sets;
}

std::vector<CaptureSet> Capturability::getCaptureRegion(const State state, const int nstep) {
  std::vector<CaptureSet> sets;

  if (grid->existState(state) ) {
    sets = getCaptureRegion(grid->getStateIndex(state), nstep);
  }

  return sets;
}

bool Capturability::capturable(State state, int nstep) {
  int state_id = grid->getStateIndex(state);

  return capturable(state_id, nstep);
}

bool Capturability::capturable(int state_id, int nstep) {
  bool flag = false;

  if(state_id > 0) {
    if (nstep == 0) {
      if (data_basin[state_id] == 0)
        flag = true;
    } else {
      if (!getCaptureRegion(state_id, nstep).empty() )
        flag = true;
    }
  }

  return flag;
}

int Capturability::getMaxStep(){
  return max_step;
}

} // namespace Capt