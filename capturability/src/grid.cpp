#include "grid.h"

namespace Capt {

Grid::Grid(Param *param) : param(param) {
  state.clear();
  input.clear();

  state_num = 0;
  input_num = 0;

  icp_x_num = 0;
  icp_y_num = 0;
  swf_x_num = 0;
  swf_y_num = 0;
  cop_x_num = 0;
  cop_y_num = 0;

  create();
}

Grid::~Grid() {
}

void Grid::create() {
  using CaptEnum::MAX;
  using CaptEnum::MIN;
  using CaptEnum::STP;

  param->read(&icp_x[MIN], "icp_x_min");
  param->read(&icp_x[MAX], "icp_x_max");
  param->read(&icp_x[STP], "icp_x_stp");
  param->read(&icp_y[MIN], "icp_y_min");
  param->read(&icp_y[MAX], "icp_y_max");
  param->read(&icp_y[STP], "icp_y_stp");
  param->read(&swf_x[MIN], "swf_x_min");
  param->read(&swf_x[MAX], "swf_x_max");
  param->read(&swf_x[STP], "swf_x_stp");
  param->read(&swf_y[MIN], "swf_y_min");
  param->read(&swf_y[MAX], "swf_y_max");
  param->read(&swf_y[STP], "swf_y_stp");
  param->read(&cop_x[MIN], "cop_x_min");
  param->read(&cop_x[MAX], "cop_x_max");
  param->read(&cop_x[STP], "cop_x_stp");
  param->read(&cop_y[MIN], "cop_y_min");
  param->read(&cop_y[MAX], "cop_y_max");
  param->read(&cop_y[STP], "cop_y_stp");

  // num of each grid
  param->read(&icp_x_num, "icp_x_num");
  param->read(&icp_y_num, "icp_y_num");
  param->read(&swf_x_num, "swf_x_num");
  param->read(&swf_y_num, "swf_y_num");
  param->read(&cop_x_num, "cop_x_num");
  param->read(&cop_y_num, "cop_y_num");

  // current grid value
  double icp_x_ = icp_x[MIN], icp_y_ = icp_y[MIN];
  double swf_x_ = swf_x[MIN], swf_y_ = swf_y[MIN];
  double cop_x_ = icp_x[MIN], cop_y_ = icp_y[MIN];

  // state
  state_num = 0;
  for (int i = 0; i < icp_x_num; i++) {
    icp_x_ = icp_x[MIN] + icp_x[STP] * i;
    for (int j = 0; j < icp_y_num; j++) {
      icp_y_ = icp_y[MIN] + icp_y[STP] * j;
      for (int k = 0; k < swf_x_num; k++) {
        swf_x_ = swf_x[MIN] + swf_x[STP] * k;
        for (int l = 0; l < swf_y_num; l++) {
          swf_y_ = swf_y[MIN] + swf_y[STP] * l;
          setState(icp_x_, icp_y_, swf_x_, swf_y_);
        }
      }
    }
  }

  FILE *fp_state = NULL;
  if ( ( fp_state = fopen("csv/state_table.csv", "w") ) == NULL) {
    printf("Error: couldn't open state_table.csv\n");
    exit(EXIT_FAILURE);
  }
  fprintf(fp_state, "index,icp_x,icp_y,swf_x,swf_y\n");
  for (int i = 0; i < max(icp_x_num, icp_y_num, swf_x_num, swf_y_num); i++) {
    fprintf(fp_state, "%d", i);
    if (i < icp_x_num) {
      icp_x_ = icp_x[MIN] + icp_x[STP] * i;
      fprintf(fp_state, ",%lf", icp_x_);
    } else {
      fprintf(fp_state, ",");
    }
    if (i < icp_y_num) {
      icp_y_ = icp_y[MIN] + icp_y[STP] * i;
      fprintf(fp_state, ",%lf", icp_y_);
    } else {
      fprintf(fp_state, ",");
    }
    if (i < swf_x_num) {
      swf_x_ = swf_x[MIN] + swf_x[STP] * i;
      fprintf(fp_state, ",%lf", swf_x_);
    } else {
      fprintf(fp_state, ",");
    }
    if (i < swf_y_num) {
      swf_y_ = swf_y[MIN] + swf_y[STP] * i;
      fprintf(fp_state, ",%lf", swf_y_);
    } else {
      fprintf(fp_state, ",");
    }
    fprintf(fp_state, "\n");
  }
  fclose(fp_state);

  // input
  input_num = 0;
  for (int i = 0; i < cop_x_num; i++) {
    cop_x_ = cop_x[MIN] + cop_x[STP] * i;
    for (int j = 0; j < cop_y_num; j++) {
      cop_y_ = cop_y[MIN] + cop_y[STP] * j;
      for (int k = 0; k < swf_x_num; k++) {
        swf_x_ = swf_x[MIN] + swf_x[STP] * k;
        for (int l = 0; l < swf_y_num; l++) {
          swf_y_ = swf_y[MIN] + swf_y[STP] * l;
          setInput(cop_x_, cop_y_, swf_x_, swf_y_);
        }
      }
    }
  }

  FILE *fp_input = NULL;
  if ( ( fp_input = fopen("csv/input_table.csv", "w") ) == NULL) {
    printf("Error: couldn't open input_table.csv\n");
    exit(EXIT_FAILURE);
  }
  fprintf(fp_input, "index,cop_x,cop_y,swf_x,swf_y\n");
  for (int i = 0; i < max(cop_x_num, cop_y_num, swf_x_num, swf_y_num); i++) {
    fprintf(fp_input, "%d", i);
    if (i < cop_x_num) {
      cop_x_ = cop_x[MIN] + cop_x[STP] * i;
      fprintf(fp_input, ",%lf", cop_x_);
    } else {
      fprintf(fp_input, ",");
    }
    if (i < cop_y_num) {
      cop_y_ = cop_y[MIN] + cop_y[STP] * i;
      fprintf(fp_input, ",%lf", cop_y_);
    } else {
      fprintf(fp_input, ",");
    }
    if (i < swf_x_num) {
      swf_x_ = swf_x[MIN] + swf_x[STP] * i;
      fprintf(fp_input, ",%lf", swf_x_);
    } else {
      fprintf(fp_input, ",");
    }
    if (i < swf_y_num) {
      swf_y_ = swf_y[MIN] + swf_y[STP] * i;
      fprintf(fp_input, ",%lf", swf_y_);
    } else {
      fprintf(fp_input, ",");
    }
    fprintf(fp_input, "\n");
  }
  fclose(fp_input);
}

int Grid::max(int val1, int val2) {
  int max_val = 0;

  if (val1 >= val2)
    max_val = val1;
  else
    max_val = val2;

  return max_val;
}

int Grid::max(int val1, int val2, int val3, int val4) {
  int max_val = 0;

  max_val = max(max(val1, val2), max(val3, val4) );

  return max_val;
}

GridState Grid::roundState(State state_) {
  using CaptEnum::MAX;
  using CaptEnum::MIN;
  using CaptEnum::STP;

  int       state_id = -1;
  GridState gs;

  int icp_x_id = 0, icp_y_id = 0;
  int swf_x_id = 0, swf_y_id = 0;

  icp_x_id = round( ( state_.icp.x() - icp_x[MIN] ) / icp_x[STP]);
  icp_y_id = round( ( state_.icp.y() - icp_y[MIN] ) / icp_y[STP]);
  swf_x_id = round( ( state_.swf.x() - swf_x[MIN] ) / swf_x[STP]);
  swf_y_id = round( ( state_.swf.y() - swf_y[MIN] ) / swf_y[STP]);

  if(0 <= icp_x_id && icp_x_id < icp_x_num &&
     0 <= icp_y_id && icp_y_id < icp_y_num &&
     0 <= swf_x_id && swf_x_id < swf_x_num &&
     0 <= swf_y_id && swf_y_id < swf_y_num)
    state_id = getStateIndex(icp_x_id, icp_y_id, swf_x_id, swf_y_id);

  gs.id    = state_id;
  gs.state = getState(state_id);

  return gs;
}

int Grid::getStateIndex(State state_) {
  return roundState(state_).id;
}

void Grid::setState(double icp_x, double icp_y, double swf_x, double swf_y) {
  State state_(icp_x, icp_y, swf_x, swf_y);
  state.push_back(state_);
  state_num++;
}

void Grid::setInput(double cop_x, double cop_y, double swf_x, double swf_y) {
  Input input_(cop_x, cop_y, swf_x, swf_y);
  input.push_back(input_);
  input_num++;
}

bool Grid::existState(int state_id) {
  bool is_exist = false;
  if (0 <= state_id && state_id < state_num)
    is_exist = true;

  return is_exist;
}

bool Grid::existState(State state_) {
  using CaptEnum::MAX;
  using CaptEnum::MIN;
  using CaptEnum::STP;

  bool flag = false;

  bool flag_icp_x = false, flag_icp_y = false;
  bool flag_swf_x = false, flag_swf_y = false;

  // icp_x
  if (state_.icp.x() >= icp_x[MIN] - icp_x[STP] / 2.0 &&
      state_.icp.x() < icp_x[MAX] + icp_x[STP] / 2.0) {
    flag_icp_x = true;
  }
  // icp_y
  if (state_.icp.y() >= icp_y[MIN] - icp_y[STP] / 2.0 &&
      state_.icp.y() < icp_y[MAX] + icp_y[STP] / 2.0) {
    flag_icp_y = true;
  }
  // swf_x
  if (state_.swf.x() >= swf_x[MIN] - swf_x[STP] / 2.0 &&
      state_.swf.x() < swf_x[MAX] + swf_x[STP] / 2.0) {
    flag_swf_x = true;
  }
  // swf_y
  if (state_.swf.y() >= swf_y[MIN] - swf_y[STP] / 2.0 &&
      state_.swf.y() < swf_y[MAX] + swf_y[STP] / 2.0) {
    flag_swf_y = true;
  }
  flag = flag_icp_x * flag_icp_y * flag_swf_x * flag_swf_y;

  return flag;
}

bool Grid::existInput(int input_id) {
  bool is_exist = false;
  if (0 <= input_id && input_id <= input_num)
    is_exist = true;

  return is_exist;
}

int Grid::getStateIndex(int icp_x_id, int icp_y_id, int swf_x_id, int swf_y_id) {
  int index = 0;
  index = swf_y_num * swf_x_num * icp_y_num * icp_x_id +
          swf_y_num * swf_x_num * icp_y_id +
          swf_y_num * swf_x_id +
          swf_y_id;
  return index;
}

State Grid::getState(int index) {
  State state_;
  if (existState(index) ) {
    state_ = state[index];
  }else{
    state_.set(-1, -1, -1, -1);
  }

  return state_;
}

Input Grid::getInput(int index) {
  Input input_;
  if (existInput(index) ) {
    input_ = input[index];
  }else{
    input_.set(-1, -1, -1, -1);
  }
  return input_;
}

int Grid::getNumState() {
  return state_num;
}

int Grid::getNumInput() {
  return input_num;
}

int Grid::getNumGrid() {
  return state_num * input_num;
}

} // namespace Capt