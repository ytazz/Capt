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

  // initialize
  // for(int state_id = 0; state_id < state_num; state_id++) {
  //   data_basin[state_id] = -1;
  //   for(int input_id = 0; input_id < input_num; input_id++) {
  //     data_nstep[state_id][input_id].state_id = state_id;
  //     data_nstep[state_id][input_id].input_id = input_id;
  //     data_nstep[state_id][input_id].next_id  = -1;
  //     data_nstep[state_id][input_id].nstep    = -1;
  //   }
  //   printf("%1.3lf\n", 100 * (double)state_id / state_num);
  // }

  max_step = 0;
}

Capturability::~Capturability() {
}

void Capturability::loadBasin(std::string file_name) {
  FILE *fp = fopen(file_name.c_str(), "r");
  if ( fp == NULL) {
    printf("Error: Couldn't find the file \"%s\"\n", file_name.c_str() );
    exit(EXIT_FAILURE);
  }
  int id = 0;

  printf("Find Basin database.\n");

  int buf;
  while (fscanf(fp, "%d", &buf) != EOF) {
    data_basin[id] = buf;
    id++;
  }

  printf("Read success! (%d data)\n", id);
  fclose(fp);
}

void Capturability::loadNstep(std::string file_name, int n) {
  FILE *fp = fopen(file_name.c_str(), "r");
  if ( fp == NULL) {
    printf("Error: Couldn't find the file \"%s\"\n", file_name.c_str() );
    exit(EXIT_FAILURE);
  }
  int id = 0;

  printf("Find %dstep database.\n", n);

  int state_id;
  int input_id;
  int next_id;
  while (fscanf(fp, "%d,%d,%d", &state_id, &input_id, &next_id) != EOF) {
    data_nstep[state_id][input_id].state_id = state_id;
    data_nstep[state_id][input_id].input_id = input_id;
    data_nstep[state_id][input_id].next_id  = next_id;
    data_nstep[state_id][input_id].nstep    = n;

    if(max_step < n)
      max_step = n;

    id++;
  }

  printf("Read success! (%d data)\n", id);
  fclose(fp);
}

std::vector<CaptureSet*> Capturability::getCaptureRegion(const int state_id) {
  std::vector<CaptureSet*> sets;

  // sets.clear();
  for (int i = 0; i < grid->getNumInput(); i++) {
    if (data_nstep[state_id][i].nstep > 0) {
      sets.push_back(&data_nstep[state_id][i]);
    }
  }

  return sets;
}

std::vector<CaptureSet*> Capturability::getCaptureRegion(const int state_id, const int nstep) {
  std::vector<CaptureSet*> sets;

  // sets.clear();
  for (int i = 0; i < grid->getNumInput(); i++) {
    if (data_nstep[state_id][i].nstep == nstep) {
      sets.push_back(&data_nstep[state_id][i]);
    }
  }

  return sets;
}

std::vector<CaptureSet*> Capturability::getCaptureRegion(const State state, const int nstep) {
  std::vector<CaptureSet*> sets;

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